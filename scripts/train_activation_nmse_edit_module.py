#!/usr/bin/env python3
from __future__ import annotations

"""Train the standard counterfactual residual MLP with activation NMSE."""

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from behavior_pipeline.runtime_env import configure_runtime_caches  # noqa: E402
configure_runtime_caches(Path(__file__).resolve().parents[1])

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from behavior_pipeline.activation_reconstruction import ACTIVATION_NMSE_DEFINITION, ActivationReconstructionDataset, activation_recovery, token_nmse
from behavior_pipeline.artifact_identity import cache_identity  # noqa: E402
from behavior_pipeline.numerics import require_finite_gradients  # noqa: E402
from behavior_pipeline.checkpoint_contract import finalize_best_checkpoint  # noqa: E402
from behavior_pipeline.edit_modules import (  # noqa: E402
    make_edit_module,
    num_update_steps,
    save_checkpoint,
    set_edit_input_norm,
)
from behavior_pipeline.topk_kl import (  # noqa: E402
    FINAL_QUESTION_RESPONSE_POSITION_POLICY,
)


def _collate_rows(batch):
    return batch


def _direction_pairs(batch: list[dict], direction: str):
    if direction == "c_to_i":
        return [(row["control_hidden"], row["intervention_hidden"]) for row in batch]
    if direction == "i_to_c":
        return [(row["intervention_hidden"], row["control_hidden"]) for row in batch]
    raise ValueError(direction)


def _loss_for_batch(edit_module, batch: list[dict], *, device: str, directions: str) -> tuple[torch.Tensor, dict]:
    """Average token NMSE within each sequence, then weight sequences equally."""
    direction_list = ["c_to_i", "i_to_c"] if directions == "both" else [directions]
    sequence_losses: list[torch.Tensor] = []
    source_values: list[float] = []
    edited_values: list[float] = []
    for direction in direction_list:
        pairs = _direction_pairs(batch, direction)
        if not pairs:
            continue
        lengths = [int(source.shape[0]) for source, _target in pairs]
        source_cat = torch.cat([source for source, _target in pairs], dim=0).to(device=device)
        target_cat = torch.cat([target for _source, target in pairs], dim=0).to(device=device)
        sign = 1 if direction == "c_to_i" else -1
        edit = edit_module.forward_direction(source_cat.unsqueeze(0), sign)[0].float()
        prediction = source_cat.float() + edit
        token_edited = token_nmse(prediction, target_cat)
        token_source = token_nmse(source_cat, target_cat)
        offset = 0
        for row, length in zip(batch, lengths):
            edited_seq = token_edited[offset:offset + length].mean()
            source_seq = token_source[offset:offset + length].mean()
            if not bool(torch.isfinite(edited_seq) & torch.isfinite(source_seq)):
                raise RuntimeError(
                    f"non-finite activation NMSE: source_id={row.get('source_id')} "
                    f"response_id={row.get('response_id')} direction={direction}"
                )
            sequence_losses.append(edited_seq)
            edited_values.append(float(edited_seq.detach().cpu()))
            source_values.append(float(source_seq.detach().cpu()))
            offset += length
    if not sequence_losses:
        return torch.zeros((), device=device, requires_grad=True), {}
    edited = float(sum(edited_values) / len(edited_values))
    source = float(sum(source_values) / len(source_values))
    return torch.stack(sequence_losses).mean(), {
        "loss": edited,
        "nmse": edited,
        "source_nmse": source,
        "activation_recovery": activation_recovery(source, edited),
        "_sequence_count": len(sequence_losses),
        "_edited_sum": float(sum(edited_values)),
        "_source_sum": float(sum(source_values)),
    }


def _merge(rows: list[dict]) -> dict:
    count = int(sum(int(row.get("_sequence_count", 0)) for row in rows))
    if count <= 0:
        return {}
    edited_sum = float(sum(float(row.get("_edited_sum", 0.0)) for row in rows))
    source_sum = float(sum(float(row.get("_source_sum", 0.0)) for row in rows))
    edited = edited_sum / count
    source = source_sum / count
    if not math.isfinite(edited) or not math.isfinite(source):
        raise RuntimeError("non-finite activation NMSE aggregate")
    return {
        "loss": edited,
        "nmse": edited,
        "source_nmse": source,
        "activation_recovery": activation_recovery(source, edited),
        "_sequence_count": count,
        "_edited_sum": edited_sum,
        "_source_sum": source_sum,
    }


@torch.no_grad()
def evaluate_nmse(edit_module, loader, *, device: str, directions: str = "both") -> dict:
    edit_module.eval()
    rows = []
    for batch in loader:
        _loss, metrics = _loss_for_batch(edit_module, batch, device=device, directions=directions)
        rows.append(metrics)
    merged = _merge(rows)
    return {key: merged[key] for key in ("loss", "nmse", "source_nmse", "activation_recovery") if key in merged}


def _gradient_global_norm(parameters) -> float:
    squares = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt().cpu())


def _normalize_sequence_gradients(parameters, sequence_count: int) -> None:
    """Normalize accumulated sequence-loss sums by the exact sequence count."""
    if int(sequence_count) <= 0:
        raise ValueError("sequence_count must be positive")
    scale = float(sequence_count)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(scale)


def _save_plots(out_dir: Path, train_history: list[dict], eval_history: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    steps = [int(row["update_step"]) for row in train_history]
    fig, ax = plt.subplots()
    if steps:
        ax.plot(steps, [float(row["nmse"]) for row in train_history], linewidth=1.0, label="train")
    for split, marker in (("train_eval", "s"), ("val", "o")):
        subset = [row for row in eval_history if row.get("split") == split]
        if subset:
            ax.plot([int(row["update_step"]) for row in subset], [float(row["nmse"]) for row in subset], marker=marker, label=split)
    ax.set_xlabel("Optimizer update")
    ax.set_ylabel("Normalized MSE")
    ax.set_title("Activation reconstruction training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_nmse_curve.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--warmup_fraction", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--eval_batch_size", type=int, default=4)
    ap.add_argument("--grad_accum_steps", type=int, default=16)
    ap.add_argument("--log_every_updates", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_eval_fraction", type=float, default=0.1)
    ap.add_argument("--directions", choices=["both", "c_to_i", "i_to_c"], default="both")
    ap.add_argument("--input_norm_mode", choices=["rms"], default="rms")
    ap.add_argument("--model_name", default="", help="Recorded for checkpoint provenance; the frozen LM is not needed during cached activation training")
    ap.add_argument("--layer", type=int, default=20)
    args = ap.parse_args()

    if not 0.0 <= args.warmup_fraction < 1.0:
        ap.error("--warmup_fraction must be in [0,1)")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ds = ActivationReconstructionDataset(args.cache_dir, "train")
    val_ds = ActivationReconstructionDataset(args.cache_dir, "val")
    d_model = int(train_ds.meta["d_model"])
    if int(val_ds.meta["d_model"]) != d_model:
        raise RuntimeError("activation cache d_model differs across splits")
    if int(train_ds.meta.get("layer", args.layer)) != int(args.layer):
        raise RuntimeError("requested layer does not match activation cache")
    if args.model_name and str(train_ds.meta.get("model_name")) != str(args.model_name):
        raise RuntimeError("requested model does not match activation cache")
    if str(train_ds.meta.get("position_policy", "")) != FINAL_QUESTION_RESPONSE_POSITION_POLICY:
        raise RuntimeError(
            "activation training requires a cache covering the final question and response tokens"
        )

    train_eval_indices, train_eval_source_ids = train_ds.source_subset_indices(
        fraction=float(args.train_eval_fraction), seed=int(args.seed)
    )
    train_eval_ds = Subset(train_ds, train_eval_indices)
    edit_module = make_edit_module(
        "base_mlp", d_model, intermediate_size=d_model,
        activation_name="silu", gated_mlp=True,
    ).to(args.device)
    set_edit_input_norm(edit_module, "rms")
    optimizer = torch.optim.Adam(
        [p for p in edit_module.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=0.0,
    )
    total_updates = max(1, int(args.epochs) * num_update_steps(len(train_ds), batch_size=args.batch_size, grad_accum_steps=args.grad_accum_steps))
    warmup_updates = int(math.ceil(args.warmup_fraction * total_updates)) if args.warmup_fraction > 0 else 0

    loaders = {
        "val": DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=_collate_rows),
    }
    if train_eval_indices:
        loaders["train_eval"] = DataLoader(train_eval_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=_collate_rows)

    checkpoint_config = {
        "cache_dir": str(args.cache_dir),
        "model_name": str(args.model_name or train_ds.meta.get("model_name", "")),
        "layer": int(args.layer),
        "edit_kind": "base_mlp",
        "objective": "activation_nmse",
        "nmse_definition": ACTIVATION_NMSE_DEFINITION,
        "activation_position_policy": str(train_ds.meta.get("position_policy")),
        "input_norm_mode": "rms",
        "directions": str(args.directions),
        "lr": float(args.lr),
        "warmup_fraction": float(args.warmup_fraction),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "grad_accum_steps": int(args.grad_accum_steps),
        "train_eval_fraction": float(args.train_eval_fraction),
        "seed": int(args.seed),
        "initialization": "zero_write_he_upstream",
        "edit_mlp_hidden_size": d_model,
        "edit_mlp_activation": "silu",
        "edit_mlp_gated": True,
        "edit_mlp_architecture": "swiglu_dmodel",
        "edit_mlp_upstream_initialization": "he_normal_fan_in_relu",
        "edit_mlp_output_initialization": "zeros",
    }

    train_history: list[dict] = []
    eval_history: list[dict] = []
    global_step = 0
    update_step = 0
    best = {"val_nmse": float("inf"), "epoch": None}

    for epoch in range(1, args.epochs + 1):
        edit_module.train()
        loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=train_ds.shard_shuffled_indices(seed=int(args.seed) + epoch),
            collate_fn=_collate_rows,
        )
        optimizer.zero_grad(set_to_none=True)
        accum_rows: list[dict] = []
        accumulated_sequences = 0
        for batch_idx, batch in enumerate(loader, start=1):
            loss, metrics = _loss_for_batch(edit_module, batch, device=args.device, directions=args.directions)
            if not torch.isfinite(loss.detach()):
                raise RuntimeError("non-finite activation NMSE training loss")
            sequence_count = int(metrics.get("_sequence_count", 0))
            (loss * float(sequence_count)).backward()
            accumulated_sequences += sequence_count
            accum_rows.append(metrics)
            global_step += 1
            should_step = (batch_idx % args.grad_accum_steps == 0) or batch_idx == len(loader)
            if not should_step:
                continue
            _normalize_sequence_gradients(edit_module.parameters(), accumulated_sequences)
            grad_norm = _gradient_global_norm(edit_module.parameters())
            next_update = update_step + 1
            lr = float(args.lr) * float(next_update) / float(warmup_updates) if warmup_updates > 0 and next_update <= warmup_updates else float(args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            require_finite_gradients(edit_module.named_parameters())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update_step = next_update
            merged = _merge(accum_rows)
            row = {
                "epoch": epoch, "global_step": global_step, "update_step": update_step,
                "learning_rate": lr, "gradient_global_norm": grad_norm,
                "accumulated_sequence_directions": accumulated_sequences,
                **{key: merged[key] for key in ("loss", "nmse", "source_nmse", "activation_recovery")},
                **edit_module.diagnostics(),
            }
            train_history.append(row)
            if args.log_every_updates > 0 and (update_step == 1 or update_step % args.log_every_updates == 0):
                print(json.dumps({"activation_nmse_train_update": row}), flush=True)
            accum_rows = []
            accumulated_sequences = 0

        if "train_eval" in loaders:
            metrics = evaluate_nmse(edit_module, loaders["train_eval"], device=args.device, directions=args.directions)
            row = {"split": "train_eval", "epoch": epoch, "global_step": global_step, "update_step": update_step, "source_count": len(train_eval_source_ids), **metrics}
            eval_history.append(row)
            print(json.dumps({"activation_nmse_eval": row}), flush=True)
        metrics = evaluate_nmse(edit_module, loaders["val"], device=args.device, directions=args.directions)
        row = {"split": "val", "epoch": epoch, "global_step": global_step, "update_step": update_step, **metrics}
        eval_history.append(row)
        print(json.dumps({"activation_nmse_eval": row}), flush=True)
        if float(metrics["nmse"]) < float(best["val_nmse"]):
            best = {"val_nmse": float(metrics["nmse"]), "epoch": epoch}
            save_checkpoint(out_dir / "model.pt", edit_module, config=checkpoint_config, metrics=best)

    if best["epoch"] is None:
        raise RuntimeError("no trained epoch was available for activation-NMSE checkpoint selection")
    save_checkpoint(out_dir / "last_model.pt", edit_module, config=checkpoint_config, metrics={"epoch": args.epochs})
    trainable_parameters = int(sum(p.numel() for p in edit_module.parameters() if p.requires_grad))
    base_metrics = {
        **checkpoint_config,
        **best,
        "architecture": "swiglu_dmodel",
        "trainable_parameters": trainable_parameters,
        "validation_source_splits": ["val"],
        "train_eval_source_count": len(train_eval_source_ids),
        "activation_cache_identity": cache_identity(args.cache_dir),
    }
    final = finalize_best_checkpoint(
        out_dir / "model.pt",
        edit_module,
        selected_epoch=int(best["epoch"]),
        base_metrics=base_metrics,
        evaluate_best=lambda module: evaluate_nmse(module.to(args.device), loaders["val"], device=args.device, directions=args.directions),
    )
    (out_dir / "train_history.json").write_text(json.dumps(train_history, indent=2), encoding="utf-8")
    (out_dir / "eval_history.json").write_text(json.dumps(eval_history, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    _save_plots(out_dir, train_history, eval_history)


if __name__ == "__main__":
    main()
