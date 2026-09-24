#!/usr/bin/env python3
from __future__ import annotations

"""Run the activation-NMSE learning-rate sweep for the standard residual MLP."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from behavior_pipeline.runtime_env import configure_runtime_caches  # noqa: E402
configure_runtime_caches(Path(__file__).resolve().parents[1])

from behavior_pipeline.activation_training_harness import (  # noqa: E402
    load_activation_training_config,
    run_activation_training_sweep,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", "--devices", dest="devices", nargs="+", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_activation_training_config(args.config)
    devices = None
    if args.devices:
        devices = [piece.strip() for value in args.devices for piece in str(value).split(",") if piece.strip()]
    manifest = run_activation_training_sweep(config, root=root, devices=devices, force=args.force, dry_run=args.dry_run)
    print(
        f"[activation-training] LR sweep complete: {len(manifest['runs'])} runs; "
        f"selected={len(manifest.get('selected', []))} by validation NMSE; "
        f"manifest={config['run']['output_root']}/training_manifest.json"
    )


if __name__ == "__main__":
    main()
