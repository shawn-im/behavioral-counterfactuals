#!/usr/bin/env python3
from __future__ import annotations

"""Train the counterfactual edit module and lightweight sanity checks on one or more GPUs."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from behavior_pipeline.runtime_env import configure_runtime_caches  # noqa: E402

configure_runtime_caches(Path(__file__).resolve().parents[1])

from behavior_pipeline.training_harness import (  # noqa: E402
    load_training_harness_config,
    run_training_sweep,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--device",
        "--devices",
        dest="devices",
        nargs="+",
        help="Override the YAML device list for this invocation",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain completed variants",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Write the expanded manifest without launching training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_training_harness_config(Path(args.config))
    devices = None
    if args.devices:
        devices = [
            device.strip()
            for value in args.devices
            for device in str(value).split(",")
            if device.strip()
        ]
    manifest = run_training_sweep(
        config,
        root=root,
        devices=devices,
        force=bool(args.force),
        dry_run=bool(args.dry_run),
    )
    selected = manifest.get("selected", [])
    print(
        f"[training] LR sweep complete: {len(manifest['runs'])} runs; "
        f"selected={len(selected)} methods by validation recovery; "
        "no test evaluation was run; "
        f"manifest={config['run']['output_root']}/training_manifest.json"
    )


if __name__ == "__main__":
    main()
