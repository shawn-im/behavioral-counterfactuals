#!/usr/bin/env python3
from __future__ import annotations

"""Train activation edit modules with off-policy conditional top-k KL distillation."""

import argparse
import gc
import json
import math
import random
import sys
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from behavior_pipeline.runtime_env import configure_runtime_caches  # noqa: E402

configure_runtime_caches(Path(__file__).resolve().parents[1])

import numpy as np
import torch
from torch.utils.data import DataLoader

from behavior_pipeline.edit_modules import (  # noqa: E402
    EditPatchHook,
    make_edit_module,
    num_update_steps,
    save_checkpoint,
    set_edit_input_norm,
)
from behavior_pipeline.checkpoint_contract import finalize_best_checkpoint  # noqa: E402
from behavior_pipeline.training_harness import fixed_source_eval_rows  # noqa: E402
from behavior_pipeline.numerics import require_finite_gradients  # noqa: E402
from behavior_pipeline.model_utils import load_causal_lm, resolve_hidden_size  # noqa: E402
from behavior_pipeline.support_projection import (  # noqa: E402
    BatchedSupportOnlyOutputHead,
)
from behavior_pipeline.topk_kl import ALL_TOKENS_POSITION_POLICY, FINAL_QUESTION_RESPONSE_POSITION_POLICY, TopKKLDistillDataset, aggregate_evaluation_rows, cached_no_edit_evaluation, conditional_logp_from_support_logits, conditional_topk_kl_raw, final_question_start_index, logp_on_support_from_logits, normalize_sequence_gradients, prompt_ids_for_direction, recovery_fraction, summarize_support_masses, source_logps_for_direction, target_logps_for_direction


TRAINING_SPLITS = ("train", "val")


def _emit_event(event: str, payload: dict) -> None:
    print(json.dumps({event: payload}), flush=True)


def _collate_rows(batch):
    return batch


def _tensor_shape(x) -> tuple[int, ...]:
    return tuple(int(v) for v in getattr(x, "shape", ()))


def _flatten_metadata(meta: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in meta.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten_metadata(value, name))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[name] = value
    return out


def _validate_cache_metadata(meta: dict, args: argparse.Namespace) -> dict:
    """Validate assumptions this trainer cannot infer from tensors alone.

    Validate available metadata fields; omitted optional fields are allowed.
    Reject incompatible temperatures and model identities unless the caller
    explicitly permits a model-name mismatch.
    """
    warnings = []
    flat = _flatten_metadata(meta)
    for key in ("temperature", "distill_temperature", "teacher_temperature", "cache_temperature"):
        if key in flat and flat[key] is not None:
            try:
                temp = float(flat[key])
            except (TypeError, ValueError):
                warnings.append(f"metadata {key}={flat[key]!r} is not numeric")
                continue
            if abs(temp - 1.0) > 1e-8:
                raise ValueError(
                    f"cache was built with {key}={temp}, but this trainer assumes T=1.0; rebuild the cache or add explicit temperature support"
                )
    model_name = flat.get("model_name") or flat.get("model")
    if model_name is not None and str(model_name) != str(args.model_name):
        message = (
            f"cache model_name={model_name!r} differs from trainer "
            f"model_name={args.model_name!r}"
        )
        if not bool(args.allow_cache_model_mismatch):
            raise ValueError(message)
        warnings.append("allowed model mismatch: " + message)
    support_mode = flat.get("support_mode") or flat.get("topk_support_mode")
    if support_mode is not None and "union" not in str(support_mode).lower():
        warnings.append(f"cache support_mode={support_mode!r}; expected deterministic top-k target/source union support")
    top_k = flat.get("top_k") or flat.get("k")
    out = {
        "cache_top_k": top_k,
        "cache_support_mode": support_mode,
        "cache_model_name": model_name,
        "cache_metadata_warnings": warnings,
    }
    return out


def _check_no_duplicate_masked_ids(support_ids: torch.Tensor, support_mask: torch.Tensor, *, row_idx: int, split: str) -> None:
    # support_ids: [T, K], support_mask: [T, K]
    for t in range(int(support_ids.shape[0])):
        ids = support_ids[t][support_mask[t]].tolist()
        if len(ids) != len(set(int(x) for x in ids)):
            raise ValueError(f"duplicate support ids in split={split} row={row_idx} token={t}")


def _validate_cache_dataset(ds: TopKKLDistillDataset, *, split: str, vocab_size: int, max_rows: int) -> dict:
    required = [
        "control_prompt_ids",
        "intervention_prompt_ids",
        "response_ids",
        "support_ids",
        "support_mask",
        "logp_control",
        "logp_control_other",
        "logp_intervention",
        "logp_intervention_other",
        "logp_control_other",
        "logp_intervention_other",
    ]
    rows_to_check = ds.rows if int(max_rows) <= 0 else ds.rows[: int(max_rows)]
    for row_idx, row in enumerate(rows_to_check):
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"cache split={split} row={row_idx} missing keys: {missing}")
        response_len = int(row["response_ids"].numel())
        support_ids = row["support_ids"]
        support_mask = row["support_mask"].bool()
        if support_ids.ndim != 2:
            raise ValueError(f"cache split={split} row={row_idx} support_ids must be [T,K], got {_tensor_shape(support_ids)}")
        if support_mask.shape != support_ids.shape:
            raise ValueError(f"cache split={split} row={row_idx} support_mask shape {_tensor_shape(support_mask)} != support_ids shape {_tensor_shape(support_ids)}")
        if int(support_ids.shape[0]) != response_len:
            raise ValueError(f"cache split={split} row={row_idx} response_len={response_len} but support T={support_ids.shape[0]}")
        for key in ("logp_control", "logp_intervention"):
            if row[key].shape != support_ids.shape:
                raise ValueError(f"cache split={split} row={row_idx} {key} shape {_tensor_shape(row[key])} != support_ids shape {_tensor_shape(support_ids)}")
            masked_vals = row[key][support_mask].float()
            if masked_vals.numel() and not torch.isfinite(masked_vals).all():
                raise ValueError(f"cache split={split} row={row_idx} {key} has nonfinite masked support logprobs")
        for key in ("logp_control_other", "logp_intervention_other"):
            if row[key].shape != (response_len,):
                raise ValueError(f"cache split={split} row={row_idx} {key} shape {_tensor_shape(row[key])} != ({response_len},)")
            if not torch.isfinite(row[key].float()).all():
                raise ValueError(f"cache split={split} row={row_idx} {key} has nonfinite values")
        valid_ids = support_ids[support_mask]
        if valid_ids.numel():
            if int(valid_ids.min()) < 0 or int(valid_ids.max()) >= int(vocab_size):
                raise ValueError(f"cache split={split} row={row_idx} support ids out of vocab range [0,{vocab_size})")
        _check_no_duplicate_masked_ids(support_ids, support_mask, row_idx=row_idx, split=split)
    return {f"{split}_cache_checked_rows": len(rows_to_check)}


def _student_logps_for_items(
    model_lm,
    layer_module,
    edit_module,
    items: list[tuple[dict, str]],
    *,
    device: str,
    pad_token_id: int,
    support_only_projection: bool = False,
    support_projection_chunk_size: int = 2048,
    tokenizer=None,
    edit_position_policy: str = ALL_TOKENS_POSITION_POLICY,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    seqs = []
    prompt_lens = []
    response_lens = []
    signs = []
    edit_starts = []
    if edit_position_policy not in {
        ALL_TOKENS_POSITION_POLICY,
        FINAL_QUESTION_RESPONSE_POSITION_POLICY,
    }:
        raise ValueError(f"unknown edit position policy: {edit_position_policy!r}")
    if edit_position_policy == FINAL_QUESTION_RESPONSE_POSITION_POLICY and tokenizer is None:
        raise ValueError("final-question edit masking requires a tokenizer")
    for row, direction in items:
        prompt_ids = prompt_ids_for_direction(row, direction)
        response_ids = row["response_ids"]
        seq = torch.cat([prompt_ids, response_ids]).to(dtype=torch.long)
        seqs.append(seq)
        prompt_lens.append(int(prompt_ids.numel()))
        response_lens.append(int(response_ids.numel()))
        signs.append(1 if direction == "c_to_i" else -1)
        edit_starts.append(
            0
            if edit_position_policy == ALL_TOKENS_POSITION_POLICY
            else final_question_start_index(tokenizer, prompt_ids)
        )
    if not seqs:
        return []
    max_len = max(int(seq.numel()) for seq in seqs)
    input_ids = torch.full((len(seqs), max_len), int(pad_token_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
    for idx, seq in enumerate(seqs):
        seq = seq.to(device=device)
        input_ids[idx, : int(seq.numel())] = seq
        attention_mask[idx, : int(seq.numel())] = 1
    edit_position_mask = None
    if edit_position_policy != ALL_TOKENS_POSITION_POLICY:
        edit_position_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        for idx, (start, seq) in enumerate(zip(edit_starts, seqs)):
            edit_position_mask[idx, int(start) : int(seq.numel())] = True
    sign_t = torch.tensor(signs, dtype=torch.float32, device=device)
    projection = None
    if support_only_projection:
        output_embeddings = model_lm.get_output_embeddings()
        projection = BatchedSupportOnlyOutputHead(
            output_embeddings,
            [row["support_ids"] for row, _direction in items],
            prompt_lengths=prompt_lens,
            response_lengths=response_lens,
            chunk_positions=int(support_projection_chunk_size),
        )
    projection_context = projection if projection is not None else nullcontext()
    with EditPatchHook(
        layer_module,
        edit_module,
        sign_t,
        position_mask=edit_position_mask,
    ), projection_context:
        logits = model_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
    out = []
    for idx, (row, _direction) in enumerate(items):
        prompt_len = int(prompt_lens[idx])
        response_len = int(response_lens[idx])
        if support_only_projection:
            logits_f = logits[idx, :response_len, :].float()
        else:
            logits_f = logits[
                idx, prompt_len - 1 : prompt_len - 1 + response_len, :
            ].float()
        if logits_f.shape[0] != response_len:
            raise RuntimeError(f"student logits T={logits_f.shape[0]} does not match response_len={response_len}")
        support_ids = row["support_ids"].to(device=device, dtype=torch.long)
        support_mask = row["support_mask"].to(device=device, dtype=torch.bool)
        if support_ids.shape[0] != response_len:
            raise RuntimeError(f"support T={support_ids.shape[0]} does not match response_len={response_len}")
        if support_only_projection:
            width = int(support_ids.shape[1])
            logits_f = logits_f[:, :width]
            try:
                logq_support = conditional_logp_from_support_logits(logits_f, support_mask)
            except RuntimeError as error:
                raise RuntimeError(
                    f"source_id={row.get('source_id')} response_id={row.get('response_id')} "
                    f"direction={_direction}: {error}"
                ) from error
            # Conditional-support training does not need the full-vocabulary
            # normalizer. Genuine support mass is measured by full evaluation.
            student_support_mass = torch.empty(0, device=device)
        else:
            safe_support_ids = support_ids.masked_fill(~support_mask, 0)
            if support_mask.any():
                masked_ids = safe_support_ids[support_mask]
                vocab = int(logits_f.shape[-1])
                if int(masked_ids.min()) < 0 or int(masked_ids.max()) >= vocab:
                    raise RuntimeError(f"support id out of range for vocab size {vocab}")
            try:
                logq_support, student_support_mass = logp_on_support_from_logits(
                    logits_f, safe_support_ids, support_mask
                )
            except RuntimeError as error:
                raise RuntimeError(
                    f"source_id={row.get('source_id')} response_id={row.get('response_id')} "
                    f"direction={_direction}: {error}"
                ) from error
        out.append((logq_support, support_mask, student_support_mass))
    return out


def _loss_for_batch(
    model_lm,
    layer_module,
    edit_module,
    batch,
    *,
    device: str,
    directions: str,
    forward_batch_size: int,
    pad_token_id: int,
    support_only_projection: bool = False,
    support_projection_chunk_size: int = 2048,
    backward_per_chunk: bool = False,
    split_forward_on_oom: bool = True,
    tokenizer=None,
    edit_position_policy: str = ALL_TOKENS_POSITION_POLICY,
) -> tuple[torch.Tensor, dict]:
    """Evaluate one row batch, optionally freeing each chunk graph immediately.

    Streamed backward accumulates sums of equally weighted sequence-direction
    losses. The caller divides the resulting gradients by the exact accumulated
    sequence count immediately before the optimizer step.
    """
    losses = []
    rows = []
    direction_list = ["c_to_i", "i_to_c"] if str(directions) == "both" else [str(directions)]
    items = [(row, direction) for row in batch for direction in direction_list]
    chunk_size = max(1, int(forward_batch_size))
    pending = [
        items[start : start + chunk_size]
        for start in range(0, len(items), chunk_size)
    ]
    while pending:
        chunk = pending.pop(0)
        retry_chunks = None
        try:
            student_outputs = _student_logps_for_items(
                model_lm,
                layer_module,
                edit_module,
                chunk,
                device=device,
                pad_token_id=pad_token_id,
                support_only_projection=bool(support_only_projection),
                support_projection_chunk_size=int(support_projection_chunk_size),
                tokenizer=tokenizer,
                edit_position_policy=str(edit_position_policy),
            )
        except RuntimeError as exc:
            is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or (
                "out of memory" in str(exc).lower()
            )
            if not is_oom or not split_forward_on_oom or len(chunk) <= 1:
                raise
            midpoint = max(1, len(chunk) // 2)
            left = chunk[:midpoint]
            right = chunk[midpoint:]
            retry_chunks = (left, right)
        if retry_chunks is not None:
            left, right = retry_chunks
            # Cleanup happens after leaving the exception handler so its
            # traceback no longer retains tensors from the failed forward.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                "[training] CUDA OOM in forward chunk "
                f"size={len(chunk)}; released cache and retrying as "
                f"{len(left)}+{len(right)}",
                flush=True,
            )
            pending = [left, right, *pending]
            continue
        chunk_losses = []
        for (row, direction), (student_logp, support_mask, student_support_mass) in zip(chunk, student_outputs):
            target_logp, _target_other_logp = target_logps_for_direction(row, direction)
            source_logp, _source_other_logp = source_logps_for_direction(row, direction)
            target_logp = target_logp.to(device=device)
            source_logp = source_logp.to(device=device)
            try:
                token_kl_raw = conditional_topk_kl_raw(target_logp, student_logp, support_mask)
                source_kl_raw = conditional_topk_kl_raw(target_logp, source_logp, support_mask)
            except RuntimeError as error:
                raise RuntimeError(
                    f"source_id={row.get('source_id')} response_id={row.get('response_id')} "
                    f"direction={direction}: {error}"
                ) from error
            token_kl = token_kl_raw.clamp_min(0.0)
            source_kl = source_kl_raw.clamp_min(0.0)
            sequence_loss = token_kl.mean()
            chunk_losses.append(sequence_loss)
            student_support_mass = student_support_mass.detach()
            rows.append(
                {
                    "source_id": str(row.get("source_id", "")),
                    "response_id": str(row.get("response_id", "")),
                    "direction": str(direction),
                    "loss": float(token_kl.mean().detach().cpu()),
                    "source_loss": float(source_kl.mean().detach().cpu()),
                    "student_support_mass": student_support_mass.detach().float().cpu(),
                }
            )
        if backward_per_chunk and chunk_losses:
            chunk_loss_sum = torch.stack(chunk_losses).sum()
            if not torch.isfinite(chunk_loss_sum.detach()):
                raise RuntimeError("non-finite loss in streamed backward chunk")
            chunk_loss_sum.backward()
            losses.extend(loss.detach() for loss in chunk_losses)
        else:
            losses.extend(chunk_losses)
    if not losses:
        return torch.zeros((), device=device, requires_grad=True), {}
    return torch.stack(losses).mean(), aggregate_evaluation_rows(rows)


def _merge_eval_aggregates(rows: list[dict]) -> dict:
    if not rows:
        return {}
    sequence_count = int(sum(row.get("_sequence_count", 0) for row in rows))
    if sequence_count <= 0:
        return {}
    edited = float(sum(row.get("_loss_sum", 0.0) for row in rows)) / sequence_count
    source = float(sum(row.get("_source_loss_sum", 0.0) for row in rows)) / sequence_count
    masses = torch.cat(
        [row["_student_support_mass"].reshape(-1) for row in rows]
    )
    support_summary = summarize_support_masses(masses)
    return {
        "loss": edited,
        "source_kl_mean": source,
        "edited_kl_mean": edited,
        "recovery": recovery_fraction(source, edited),
        "support_mass_p10": support_summary["p10"],
        "support_mass_min": support_summary["minimum"],
        "support_mass_histogram": support_summary["histogram"],
        "_sequence_count": sequence_count,
        "_loss_sum": edited * sequence_count,
        "_source_loss_sum": source * sequence_count,
        "_student_support_mass": masses,
        "_sequence_rows": [
            row
            for aggregate in rows
            for row in aggregate.get("_sequence_rows", [])
        ],
    }


def _public_metrics(metrics: dict, *, include_sequence_rows: bool = False) -> dict:
    keep = {
        "loss",
        "source_kl_mean",
        "edited_kl_mean",
        "recovery",
        "support_mass_p10",
        "support_mass_min",
        "support_mass_histogram",
    }
    result = {key: metrics[key] for key in keep if key in metrics}
    if include_sequence_rows:
        result["sequence_rows"] = list(metrics.get("_sequence_rows", []))
    return result


@torch.no_grad()
def evaluate(
    model_lm,
    layer_module,
    edit_module,
    loader,
    *,
    device: str,
    directions: str,
    forward_batch_size: int,
    pad_token_id: int,
    include_sequence_rows: bool = False,
    tokenizer=None,
    edit_position_policy: str = ALL_TOKENS_POSITION_POLICY,
) -> dict:
    edit_module.eval()
    rows = []
    for batch in loader:
        _loss, metrics = _loss_for_batch(
            model_lm,
            layer_module,
            edit_module,
            batch,
            device=device,
            directions=directions,
            forward_batch_size=int(forward_batch_size),
            pad_token_id=int(pad_token_id),
            tokenizer=tokenizer,
            edit_position_policy=str(edit_position_policy),
        )
        rows.append(metrics)
    if not rows:
        return {}
    return _public_metrics(
        _merge_eval_aggregates(rows),
        include_sequence_rows=bool(include_sequence_rows),
    )


def _save_training_plots(
    out_dir: Path,
    train_history: list[dict],
    eval_history: list[dict],
) -> None:
    """Write lightweight train/validation diagnostics for the current run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _rolling(values: list[float], window: int = 25) -> list[float]:
        if not values:
            return []
        result = []
        total = 0.0
        queue = []
        for value in values:
            value = float(value)
            queue.append(value)
            total += value
            if len(queue) > window:
                total -= queue.pop(0)
            result.append(total / float(len(queue)))
        return result

    train_steps = [int(row["update_step"]) for row in train_history]
    val_rows = [row for row in eval_history if row.get("split") == "val"]
    val_steps = [int(row.get("update_step", 0)) for row in val_rows]
    train_eval_rows = [
        row for row in eval_history if row.get("split") == "train_eval"
    ]
    train_eval_steps = [
        int(row.get("update_step", 0)) for row in train_eval_rows
    ]

    for metric, ylabel, filename in (
        ("loss", "Conditional KL", "training_loss_curve.png"),
        ("recovery", "Recovery", "training_recovery_curve.png"),
    ):
        fig, ax = plt.subplots()
        train_values = [float(row[metric]) for row in train_history]
        if train_steps:
            ax.plot(
                train_steps,
                train_values,
                linewidth=0.8,
                alpha=0.28,
                label="train (per update)",
            )
            ax.plot(
                train_steps,
                _rolling(train_values),
                linewidth=1.5,
                label="train (25-update rolling mean)",
            )
        if train_eval_rows:
            ax.plot(
                train_eval_steps,
                [float(row[metric]) for row in train_eval_rows],
                marker="s",
                label="train eval (fixed source subset)",
            )
        if val_rows:
            ax.plot(
                val_steps,
                [float(row[metric]) for row in val_rows],
                marker="o",
                label="validation",
            )
        if metric == "recovery":
            ax.axhline(0.0, linewidth=1.0)
        ax.set_xlabel("Optimizer update")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Training {metric}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)

    if train_steps and any("gradient_global_norm" in row for row in train_history):
        fig, ax = plt.subplots()
        grad_values = [float(row.get("gradient_global_norm", float("nan"))) for row in train_history]
        ax.plot(train_steps, grad_values, linewidth=1.0)
        ax.set_yscale("log")
        ax.set_xlabel("Optimizer update")
        ax.set_ylabel("Global gradient norm")
        ax.set_title("Training gradient norm")
        fig.tight_layout()
        fig.savefig(out_dir / "training_gradient_norm_curve.png", dpi=180)
        plt.close(fig)

    if train_steps and any("learning_rate" in row for row in train_history):
        fig, ax = plt.subplots()
        lr_values = [float(row.get("learning_rate", float("nan"))) for row in train_history]
        ax.plot(train_steps, lr_values, linewidth=1.0)
        ax.set_xlabel("Optimizer update")
        ax.set_ylabel("Learning rate")
        ax.set_title("Training learning-rate schedule")
        fig.tight_layout()
        fig.savefig(out_dir / "training_learning_rate_curve.png", dpi=180)
        plt.close(fig)


def _gradient_global_norm(parameters) -> float:
    squares = []
    for parameter in parameters:
        if parameter.grad is not None:
            squares.append(parameter.grad.detach().float().pow(2).sum())
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt().cpu())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument(
        "--model_name",
        default="google/gemma-4-E4B-it",
    )
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument(
        "--edit_kind",
        choices=["fixed_steering", "directional_linear", "base_mlp"],
        default="base_mlp",
    )
    ap.add_argument(
        "--input_norm_mode",
        choices=["rms"],
        default="rms",
    )
    ap.add_argument("--directions", choices=["both", "c_to_i", "i_to_c"], default="both")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--warmup_fraction", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--eval_batch_size", type=int, default=4)
    ap.add_argument(
        "--forward_batch_size",
        type=int,
        default=2,
        help="Number of row/direction sequences per padded LM forward",
    )
    ap.add_argument(
        "--training_support_only_projection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "During training, project the LM head only onto each token's "
            "cached conditional support. Validation remains full-vocabulary."
        ),
    )
    ap.add_argument(
        "--support_projection_chunk_size",
        type=int,
        default=2048,
        help="Response positions per support-only LM-head projection chunk.",
    )
    ap.add_argument("--grad_accum_steps", type=int, default=4)
    ap.add_argument("--log_every_updates", type=int, default=10)
    ap.add_argument(
        "--log_every_microbatches",
        type=int,
        default=0,
        help="Emit training progress every N microbatches; 0 disables it.",
    )
    ap.add_argument(
        "--train_eval_fraction",
        type=float,
        default=0.10,
        help=(
            "Fraction of training source prompts in a deterministic fixed "
            "diagnostic subset evaluated at epoch boundaries; 0 disables"
        ),
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--cache_check_max_rows",
        type=int,
        default=512,
        help="Rows per split to validate before training; <=0 checks all rows",
    )
    ap.add_argument("--skip_cache_checks", action="store_true")
    ap.add_argument(
        "--allow_cache_model_mismatch",
        action="store_true",
        help="Allow an intentionally different teacher-cache model.",
    )
    args = ap.parse_args()
    if int(args.epochs) <= 0:
        ap.error("--epochs must be positive")
    if float(args.lr) <= 0.0:
        ap.error("--lr must be positive")
    if not 0.0 <= float(args.warmup_fraction) < 1.0:
        ap.error("--warmup_fraction must be in [0, 1)")
    if not 0.0 <= float(args.train_eval_fraction) <= 1.0:
        ap.error("--train_eval_fraction must be in [0, 1]")
    if int(args.support_projection_chunk_size) <= 0:
        ap.error("--support_projection_chunk_size must be positive")
    if int(args.log_every_microbatches) < 0:
        ap.error("--log_every_microbatches must be nonnegative")

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_causal_lm(
        str(args.model_name),
        device=str(args.device),
        dtype=str(args.dtype),
    )
    model_lm = loaded.model
    tokenizer = loaded.tokenizer
    layers = loaded.layers
    for param in model_lm.parameters():
        param.requires_grad_(False)
    model_lm.eval()
    d_model = resolve_hidden_size(model_lm)

    if args.edit_kind == "base_mlp":
        print(
            "[edit-module] standard editor: "
            f"RMS -> SwiGLU {d_model}->{d_model}->{d_model}; "
            "shared gate/up, directional write heads",
            flush=True,
        )

    pad_token_id = int(
        tokenizer.pad_token_id
        if getattr(tokenizer, "pad_token_id", None) is not None
        else 0
    )
    training_datasets = {
        split: TopKKLDistillDataset(Path(args.cache_dir), split)
        for split in TRAINING_SPLITS
    }
    train_ds = training_datasets["train"]
    val_ds = training_datasets["val"]
    cache_meta_summary = _validate_cache_metadata(train_ds.meta, args)
    vocab_size = int(model_lm.get_output_embeddings().weight.shape[0])
    if not bool(args.skip_cache_checks):
        for split_name, ds in (("train", train_ds), ("val", val_ds)):
            _validate_cache_dataset(
                ds,
                split=split_name,
                vocab_size=vocab_size,
                max_rows=int(args.cache_check_max_rows),
            )

    train_eval_rows, train_eval_source_ids = fixed_source_eval_rows(
        train_ds.rows,
        fraction=float(args.train_eval_fraction),
        seed=int(args.seed),
    )

    module_kwargs = {}
    if args.edit_kind == "base_mlp":
        module_kwargs = {
            "intermediate_size": int(d_model),
            "activation_name": "silu",
            "gated_mlp": True,
        }
    edit_module = make_edit_module(
        str(args.edit_kind),
        d_model,
        **module_kwargs,
    ).to(str(args.device))
    set_edit_input_norm(edit_module, "rms")

    checkpoint_config = {
        "cache_dir": str(args.cache_dir),
        "model_name": str(args.model_name),
        "layer": int(args.layer),
        "edit_kind": str(args.edit_kind),
        "input_norm_mode": str(args.input_norm_mode),
        "directions": str(args.directions),
        "lr": float(args.lr),
        "warmup_fraction": float(args.warmup_fraction),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "forward_batch_size": int(args.forward_batch_size),
        "training_support_only_projection": bool(
            args.training_support_only_projection
        ),
        "support_projection_chunk_size": int(
            args.support_projection_chunk_size
        ),
        "backward_per_forward_chunk": True,
        "grad_accum_steps": int(args.grad_accum_steps),
        "train_eval_fraction": float(args.train_eval_fraction),
        "dtype": str(args.dtype),
        "seed": int(args.seed),
        "initialization": "zero_write_he_upstream",
        "input_global_rms": 1.0,
        "input_rms_eps": 1e-6,
    }
    if args.edit_kind == "base_mlp":
        checkpoint_config.update(
            {
                "edit_mlp_hidden_size": int(d_model),
                "edit_mlp_activation": "silu",
                "edit_mlp_gated": True,
                "edit_mlp_architecture": "swiglu_dmodel",
                "edit_mlp_upstream_initialization": "he_normal_fan_in_relu",
                "edit_mlp_output_initialization": "zeros",
            }
        )

    opt = torch.optim.Adam(
        [p for p in edit_module.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=0.0,
    )
    total_updates = max(
        1,
        int(args.epochs)
        * num_update_steps(
            len(train_ds),
            batch_size=int(args.batch_size),
            grad_accum_steps=int(args.grad_accum_steps),
        ),
    )
    warmup_updates = (
        int(math.ceil(float(args.warmup_fraction) * total_updates))
        if float(args.warmup_fraction) > 0.0
        else 0
    )

    loaders = {
        "val": DataLoader(
            val_ds,
            batch_size=int(args.eval_batch_size),
            shuffle=False,
            collate_fn=_collate_rows,
        ),
    }
    if train_eval_rows:
        loaders["train_eval"] = DataLoader(
            train_eval_rows,
            batch_size=int(args.eval_batch_size),
            shuffle=False,
            collate_fn=_collate_rows,
        )
    train_history: list[dict] = []
    eval_history: list[dict] = []
    global_step = 0
    update_step = 0
    method = str(args.edit_kind)
    trainable_parameters = int(
        sum(
            parameter.numel()
            for parameter in edit_module.parameters()
            if parameter.requires_grad
        )
    )
    architecture = (
        "base_model_style_ffn" if method == "base_mlp" else method
    )
    _emit_event(
        "topk_kl_distill_setup",
        {
            "method": method,
            "architecture": architecture,
            "trainable_parameters": trainable_parameters,
            "train_sequences": len(train_ds),
            "val_sequences": len(val_ds),
            "train_eval_fraction": float(args.train_eval_fraction),
            "train_eval_sources": len(train_eval_source_ids),
            "train_eval_sequences": len(train_eval_rows),
            "validation_source_splits": list(val_ds.source_splits),
            "total_update_steps": total_updates,
            "optimizer": "Adam",
            "learning_rate": float(args.lr),
            "warmup_fraction": float(args.warmup_fraction),
            "warmup_updates": int(warmup_updates),
            "training_support_only_projection": bool(
                args.training_support_only_projection
            ),
            "support_projection_chunk_size": int(
                args.support_projection_chunk_size
            ),
            "backward_per_forward_chunk": True,
            "nominal_rows_per_update": int(args.batch_size) * int(args.grad_accum_steps),
            "nominal_sequence_directions_per_update": int(args.batch_size) * int(args.grad_accum_steps) * (2 if str(args.directions) == "both" else 1),
            "epochs": int(args.epochs),
            "initialization": "zero_write_he_upstream",
            "base_mlp_upstream_initialization": (
                "he_normal_fan_in_relu" if method == "base_mlp" else None
            ),
            "base_mlp_output_initialization": (
                "zeros" if method == "base_mlp" else None
            ),
            "cache_dir": str(args.cache_dir),
            "input_norm_mode": str(args.input_norm_mode),
            "input_global_rms": 1.0,
            "forward_batch_size": int(args.forward_batch_size),
            "kl_objective": "conditional_topk_forward_kl_no_other_bucket",
            "vocab_size": vocab_size,
            **cache_meta_summary,
        },
    )

    # Every production editor has an exact zero-write initialization. Reuse the
    # cached source distribution instead of rerunning the frozen model twice.
    if "train_eval" in loaders:
        initial_train = cached_no_edit_evaluation(
            train_eval_rows,
            directions=str(args.directions),
        )
        initial_train_row = {
            "method": method,
            "split": "train_eval",
            "epoch": 0,
            "global_step": 0,
            "update_step": 0,
            "source_count": len(train_eval_source_ids),
            **_public_metrics(initial_train),
        }
        eval_history.append(initial_train_row)
        _emit_event("topk_kl_distill_eval", initial_train_row)

    initial_val = cached_no_edit_evaluation(
        val_ds.rows,
        directions=str(args.directions),
    )
    initial_row = {
        "method": method,
        "split": "val",
        "epoch": 0,
        "global_step": 0,
        "update_step": 0,
        **_public_metrics(initial_val),
    }
    eval_history.append(initial_row)
    _emit_event("topk_kl_distill_eval", initial_row)
    best = {"val_loss": float("inf"), "epoch": None}

    for epoch in range(1, int(args.epochs) + 1):
        edit_module.train()
        loader = DataLoader(
            train_ds,
            batch_size=int(args.batch_size),
            shuffle=True,
            collate_fn=_collate_rows,
        )
        accum_rows = []
        accumulated_sequences = 0
        opt.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(loader, start=1):
            loss, metrics = _loss_for_batch(
                model_lm,
                layers[int(args.layer)],
                edit_module,
                batch,
                device=str(args.device),
                directions=str(args.directions),
                forward_batch_size=int(args.forward_batch_size),
                pad_token_id=pad_token_id,
                support_only_projection=bool(
                    args.training_support_only_projection
                ),
                support_projection_chunk_size=int(
                    args.support_projection_chunk_size
                ),
                backward_per_chunk=True,
            )
            accum = max(1, int(args.grad_accum_steps))
            global_step += 1
            if not torch.isfinite(loss.detach()):
                raise RuntimeError(
                    "non-finite training loss at "
                    f"epoch={epoch} batch={batch_idx} global_step={global_step}"
                )
            sequence_count = int(metrics.get("_sequence_count", 0))
            if sequence_count <= 0:
                raise RuntimeError("training microbatch contains no sequences")
            accumulated_sequences += sequence_count
            accum_rows.append(metrics)
            microbatch_every = int(args.log_every_microbatches)
            if microbatch_every > 0 and (
                batch_idx == 1
                or batch_idx % microbatch_every == 0
                or batch_idx == len(loader)
            ):
                _emit_event(
                    "topk_kl_distill_train_microbatch",
                    {
                        "method": method,
                        "epoch": epoch,
                        "batch": batch_idx,
                        "batches": len(loader),
                        "global_step": global_step,
                        "update_step": update_step,
                        "microbatches_in_update": len(accum_rows),
                        "microbatches_per_update": accum,
                        "loss": float(loss.detach().cpu()),
                        "recovery": metrics.get("recovery"),
                    },
                )
            should_step = (batch_idx % accum == 0) or (batch_idx == len(loader))
            if should_step and accum_rows:
                normalize_sequence_gradients(
                    edit_module.parameters(), accumulated_sequences
                )
                grad_norm = _gradient_global_norm(edit_module.parameters())
                next_update = update_step + 1
                if warmup_updates > 0 and next_update <= warmup_updates:
                    learning_rate = float(args.lr) * float(next_update) / float(warmup_updates)
                else:
                    learning_rate = float(args.lr)
                for group in opt.param_groups:
                    group["lr"] = learning_rate
                require_finite_gradients(edit_module.named_parameters())
                opt.step()
                opt.zero_grad(set_to_none=True)
                update_step = next_update
                row = _public_metrics(_merge_eval_aggregates(accum_rows))
                row.update(
                    {
                        "method": method,
                        "epoch": epoch,
                        "global_step": global_step,
                        "update_step": update_step,
                        "learning_rate": float(learning_rate),
                        "gradient_global_norm": float(grad_norm),
                        "accumulated_sequence_directions": int(accumulated_sequences),
                        **edit_module.diagnostics(),
                    }
                )
                train_history.append(row)
                if int(args.log_every_updates) > 0 and (
                    update_step == 1
                    or update_step % int(args.log_every_updates) == 0
                ):
                    _emit_event("topk_kl_distill_train_update", row)
                accum_rows = []
                accumulated_sequences = 0

        if "train_eval" in loaders:
            train_eval_metrics = evaluate(
                model_lm,
                layers[int(args.layer)],
                edit_module,
                loaders["train_eval"],
                device=str(args.device),
                directions=str(args.directions),
                forward_batch_size=int(args.forward_batch_size),
                pad_token_id=pad_token_id,
            )
            train_eval_metrics = {
                "method": method,
                "split": "train_eval",
                "epoch": epoch,
                "global_step": global_step,
                "update_step": update_step,
                "source_count": len(train_eval_source_ids),
                **_public_metrics(train_eval_metrics),
            }
            eval_history.append(train_eval_metrics)
            _emit_event("topk_kl_distill_eval", train_eval_metrics)

        metrics = evaluate(
            model_lm,
            layers[int(args.layer)],
            edit_module,
            loaders["val"],
            device=str(args.device),
            directions=str(args.directions),
            forward_batch_size=int(args.forward_batch_size),
            pad_token_id=pad_token_id,
        )
        metrics = {
            "method": method,
            "split": "val",
            "epoch": epoch,
            "global_step": global_step,
            "update_step": update_step,
            **_public_metrics(metrics),
        }
        eval_history.append(metrics)
        _emit_event("topk_kl_distill_eval", metrics)
        if float(metrics.get("loss", float("inf"))) < best["val_loss"]:
            best = {"val_loss": float(metrics["loss"]), "epoch": epoch}
            save_checkpoint(
                out_dir / "model.pt",
                edit_module,
                config=checkpoint_config,
                metrics=best,
            )

    if best["epoch"] is None:
        raise RuntimeError("no trained epoch was available for checkpoint selection")

    save_checkpoint(
        out_dir / "last_model.pt",
        edit_module,
        config=checkpoint_config,
        metrics={"epoch": int(args.epochs)},
    )
    base_metrics = {
        **checkpoint_config,
        "architecture": architecture,
        "trainable_parameters": trainable_parameters,
        "kl_objective": "conditional_topk_forward_kl_no_other_bucket",
        **best,
        "validation_source_splits": list(val_ds.source_splits),
        "train_eval_fraction": float(args.train_eval_fraction),
        "train_eval_source_count": len(train_eval_source_ids),
        "train_eval_sequence_count": len(train_eval_rows),
    }
    final = finalize_best_checkpoint(
        out_dir / "model.pt",
        edit_module,
        selected_epoch=int(best["epoch"]),
        base_metrics=base_metrics,
        evaluate_best=lambda module: _public_metrics(
            evaluate(
                model_lm,
                layers[int(args.layer)],
                module.to(str(args.device)),
                loaders["val"],
                device=str(args.device),
                directions=str(args.directions),
                forward_batch_size=int(args.forward_batch_size),
                pad_token_id=pad_token_id,
            )
        ),
    )
    (out_dir / "train_history.json").write_text(
        json.dumps(train_history, indent=2), encoding="utf-8"
    )
    (out_dir / "eval_history.json").write_text(
        json.dumps(eval_history, indent=2), encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8"
    )
    _save_training_plots(
        out_dir, train_history, eval_history
    )


if __name__ == "__main__":
    main()
