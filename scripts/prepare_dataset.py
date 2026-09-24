#!/usr/bin/env python3
from __future__ import annotations

"""Prepare native Scruples train/dev/test splits for a supported intervention."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from behavior_pipeline.io import write_jsonl  # noqa: E402
from behavior_pipeline.runtime_env import configure_runtime_caches  # noqa: E402
from behavior_pipeline.tasks import build_scruples_examples, load_scruples_native_splits  # noqa: E402

configure_runtime_caches(Path(__file__).resolve().parents[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--intervention",
        choices=["suggest_wrong"],
        required=True,
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing Scruples train/dev/test JSONL files.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=0, help="Seed for deterministic A/B answer-label swapping.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_splits = load_scruples_native_splits(args.input_dir)
    output_dir = Path(args.output_dir)

    prepared: dict[str, list[dict]] = {}
    for split in ("train", "val", "test"):
        rows = build_scruples_examples(
            source_splits[split],
            seed=int(args.seed),
            intervention_type=str(args.intervention),
        )
        prepared[split] = rows
        write_jsonl(output_dir / f"{split}.jsonl", rows)
        print(
            f"[write] {split}: {len(rows)} rows / "
            f"{len(source_splits[split])} Scruples sources"
        )

    metadata = {
        "dataset": "scruples-anecdotes",
        "intervention": str(args.intervention),
        "source_splits": "scruples_native_train_dev_test",
        "split_mapping": {"train": "train", "dev": "val", "test": "test"},
        "seed": int(args.seed),
        "splits": {
            split: {
                "rows": len(prepared[split]),
                "sources": len(source_splits[split]),
            }
            for split in ("train", "val", "test")
        },
    }
    (output_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
