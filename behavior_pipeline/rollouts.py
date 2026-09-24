from __future__ import annotations

"""Helpers for generated paired rollout datasets."""

from pathlib import Path

from .io import load_split_dir, read_jsonl


def group_all_rollouts(rows: list[dict]) -> dict[str, dict[str, list[dict]]]:
    """Group generated rows and enforce balanced paired rollout provenance."""
    grouped: dict[str, dict[str, list[dict]]] = {}
    for idx, row in enumerate(rows):
        source_id = str(row.get("source_id") or row.get("example_id") or f"row:{idx}")
        arm = str(row.get("arm", ""))
        if arm not in {"control", "intervention"}:
            raise ValueError(f"row {idx} has invalid paired arm {arm!r}")
        grouped.setdefault(source_id, {"control": [], "intervention": []})[arm].append(row)
    rollout_counts = set()
    for source_id, arms in grouped.items():
        counts = {arm: len(values) for arm, values in arms.items()}
        if counts["control"] == 0 or counts["control"] != counts["intervention"]:
            raise ValueError(
                f"source {source_id!r} has unbalanced rollout counts: {counts}"
            )
        expected = list(range(counts["control"]))
        rollout_counts.add(counts["control"])
        for arm, values in arms.items():
            rollout_ids = sorted(int(row.get("rollout_id", -1)) for row in values)
            if rollout_ids != expected:
                raise ValueError(
                    f"source {source_id!r} arm={arm} has rollout IDs "
                    f"{rollout_ids}; expected {expected}"
                )
    if len(rollout_counts) > 1:
        raise ValueError(
            "sources have different rollout counts: "
            + ", ".join(str(value) for value in sorted(rollout_counts))
        )
    return grouped


def load_grouped_rollout_splits(data_dir: str | Path, splits: list[str] | None = None) -> dict[str, dict[str, dict[str, list[dict]]]]:
    """Load split rows and preserve every generated rollout per arm."""
    if splits is None:
        loaded = load_split_dir(data_dir)
    else:
        data_dir = Path(data_dir)
        loaded = {
            split: read_jsonl(data_dir / f"{split}.jsonl")
            for split in splits
        }
    grouped = {
        split: group_all_rollouts(rows) for split, rows in loaded.items()
    }
    owner = {}
    for split, sources in grouped.items():
        for source_id in sources:
            if source_id in owner:
                raise ValueError(
                    f"source_id {source_id!r} appears in both "
                    f"{owner[source_id]} and {split}"
                )
            owner[source_id] = split
    return grouped


def response_ids(tokenizer, response: str) -> list[int]:
    """Tokenize a fixed response as a standalone suffix."""
    return list(tokenizer(str(response), add_special_tokens=False).input_ids)
