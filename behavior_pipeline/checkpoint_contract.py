from __future__ import annotations

"""Canonical best-validation checkpoint finalization."""

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from .edit_modules import save_checkpoint


def finalize_best_checkpoint(
    path: Path,
    module: nn.Module,
    *,
    selected_epoch: int,
    base_metrics: dict,
    evaluate_best: Callable[[nn.Module], dict],
) -> dict:
    """Restore, re-evaluate, and rewrite the canonical best checkpoint."""
    payload = torch.load(path, map_location="cpu")
    module.load_state_dict(payload["state_dict"], strict=True)
    validation = evaluate_best(module)
    final = {
        **base_metrics,
        "selected_checkpoint_epoch": int(selected_epoch),
        **{f"val_{key}": value for key, value in validation.items()},
    }
    save_checkpoint(path, module, config=final, metrics=final)
    return final
