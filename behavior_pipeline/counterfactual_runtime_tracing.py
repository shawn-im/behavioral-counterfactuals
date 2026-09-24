from __future__ import annotations

"""Internal tracing implementation for :mod:`behavior_pipeline.counterfactual_runtime`."""

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .token_text import decode_token_ids, token_id_texts
from .support_projection import (
    BatchedSupportOnlyOutputHead as _BatchedSupportOnlyOutputHead,
)
from .topk_kl import (
    TopKKLDistillDataset,
    aggregate_sequence_kl,
    conditional_topk_kl,
    conditional_logp_from_support_logits,
    logp_on_support_from_logits,
    mean_token_kl,
    prompt_ids_for_direction,
    recovery_fraction,
    source_logps_for_direction,
    summarize_support_masses,
    target_logps_for_direction,
)
from .counterfactual_runtime_core import (
    Runtime,
    _FixedEditHook,
    _InlineDirectionalEditHook,
    capture_hidden,
    direction_sign,
    read_json,
    select_diverse_rows,
    target_prompt_ids,
    trace_from_hidden,
    truncate_cache_row,
    write_json,
)

_TOKEN_TEXT_CACHE: dict[tuple[int, int], str] = {}
_TOKEN_TEXT_CACHE_OWNERS: dict[int, object] = {}


def token_texts(tokenizer, token_ids: torch.Tensor) -> list[str]:
    ids = [int(value) for value in token_ids.detach().cpu().tolist()]
    decoded = token_id_texts(tokenizer, ids)
    if len(decoded) != len(ids):
        raise RuntimeError("Tokenizer returned the wrong number of token text pieces")
    values = []
    tokenizer_key = id(tokenizer)
    # Keep the tokenizer object alive for as long as entries keyed by its id are
    # cached. Otherwise Python can recycle an id for a later tokenizer and return
    # stale decoded text from a different vocabulary.
    _TOKEN_TEXT_CACHE_OWNERS[tokenizer_key] = tokenizer
    for token_id, decoded_text in zip(ids, decoded):
        key = (tokenizer_key, token_id)
        cached = _TOKEN_TEXT_CACHE.get(key)
        if cached is None:
            cached = decoded_text
            _TOKEN_TEXT_CACHE[key] = cached
        values.append(cached)
    return values


class _SupportOnlyOutputHead:
    """Temporarily replace a linear LM head with a fixed-support projection.

    The cached support is position-specific.  We project only the vocabulary
    rows in that support while returning a normal sequence-shaped logits tensor,
    so model-level postprocessing after the LM head (for example logit
    soft-capping) is preserved.  The replacement is used only by downstream IG.
    """

    def __init__(
        self,
        output_embeddings: nn.Module,
        support_ids: torch.Tensor,
        *,
        prompt_len: int,
        response_len: int,
        chunk_size: int = 64,
    ):
        if not isinstance(output_embeddings, nn.Linear):
            raise TypeError(
                "support-only IG projection currently requires a linear output head; "
                f"got {type(output_embeddings).__name__}"
            )
        self.module = output_embeddings
        self.support_ids = support_ids.detach().long()
        self.prompt_len = int(prompt_len)
        self.response_len = int(response_len)
        self.chunk_size = max(1, int(chunk_size))
        self.original_forward = None

    def _forward(self, hidden: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if args or kwargs:
            raise TypeError("support-only output projection expects hidden states only")
        if hidden.ndim != 3:
            raise ValueError(
                "support-only output projection expects [batch, sequence, hidden], "
                f"got {tuple(hidden.shape)}"
            )
        batch, sequence, hidden_size = hidden.shape
        if int(self.module.weight.shape[1]) != int(hidden_size):
            raise ValueError("LM-head hidden size does not match model hidden states")
        if int(self.support_ids.shape[0]) != self.response_len:
            raise ValueError("support rows do not match response length")
        width = int(self.support_ids.shape[1])
        first = self.prompt_len - 1
        last = first + self.response_len
        if first < 0 or last > int(sequence):
            raise ValueError(
                "response prediction positions are outside the LM-head hidden sequence: "
                f"[{first}, {last}) vs sequence={sequence}"
            )

        chunks = []
        weight = self.module.weight
        bias = self.module.bias
        for start in range(0, self.response_len, self.chunk_size):
            end = min(start + self.chunk_size, self.response_len)
            ids = self.support_ids[start:end].to(device=weight.device)
            selected_weight = weight.index_select(0, ids.reshape(-1)).reshape(
                end - start, width, hidden_size
            )
            selected_hidden = hidden[:, first + start : first + end, :]
            logits = torch.einsum("bcd,ckd->bck", selected_hidden, selected_weight)
            if bias is not None:
                selected_bias = bias.index_select(0, ids.reshape(-1)).reshape(
                    end - start, width
                )
                logits = logits + selected_bias.unsqueeze(0)
            chunks.append(logits)
        response_logits = torch.cat(chunks, dim=1)
        prefix = hidden.new_zeros((batch, first, width))
        suffix = hidden.new_zeros((batch, int(sequence) - last, width))
        return torch.cat([prefix, response_logits, suffix], dim=1)

    def __enter__(self):
        self.original_forward = self.module.forward
        self.module.forward = self._forward
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.original_forward is not None:
            self.module.forward = self.original_forward
            self.original_forward = None


class _BatchedResponseOnlyOutputHead:
    """Run the ordinary LM head only at response-prediction positions.

    Select hidden states used to predict response tokens before projecting
    through the LM head. Return their full-vocabulary logits without projecting
    unused prompt or right-padding positions.
    """

    def __init__(
        self,
        output_embeddings: nn.Module,
        *,
        prompt_lengths: list[int],
        response_lengths: list[int],
    ):
        if not isinstance(output_embeddings, nn.Linear):
            raise TypeError(
                "response-only projection requires a linear output head; "
                f"got {type(output_embeddings).__name__}"
            )
        if len(prompt_lengths) != len(response_lengths):
            raise ValueError("prompt/response batch sizes must match")
        self.module = output_embeddings
        self.prompt_lengths = [int(value) for value in prompt_lengths]
        self.response_lengths = [int(value) for value in response_lengths]
        self.original_forward = None

    def _forward(self, hidden: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.original_forward is None:
            raise RuntimeError("response-only output head is not active")
        if hidden.ndim != 3:
            raise ValueError(
                "response-only output projection expects [batch, sequence, hidden], "
                f"got {tuple(hidden.shape)}"
            )
        batch, sequence, hidden_size = hidden.shape
        if int(batch) != len(self.prompt_lengths):
            raise ValueError("LM-head batch size does not match response batch")
        max_response = max(self.response_lengths, default=0)
        compact = hidden.new_zeros((batch, max_response, hidden_size))
        for batch_index, (prompt_len, response_len) in enumerate(
            zip(self.prompt_lengths, self.response_lengths)
        ):
            first = int(prompt_len) - 1
            last = first + int(response_len)
            if first < 0 or last > int(sequence):
                raise ValueError(
                    "response prediction positions are outside the LM-head sequence: "
                    f"[{first}, {last}) vs sequence={sequence}"
                )
            compact[batch_index, :response_len] = hidden[
                batch_index, first:last
            ]
        return self.original_forward(compact, *args, **kwargs)

    def __enter__(self):
        self.original_forward = self.module.forward
        self.module.forward = self._forward
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.original_forward is not None:
            self.module.forward = self.original_forward
            self.original_forward = None


def _response_support_logits(
    runtime: Runtime,
    input_ids: torch.Tensor,
    *,
    prompt_len: int,
    response_len: int,
    edit: torch.Tensor,
    support_ids: torch.Tensor,
    chunk_size: int,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Run the model while projecting the LM head only onto cached support rows."""
    if runtime.output_embeddings is None:
        raise RuntimeError("support-only IG projection requires runtime.output_embeddings")
    context = torch.enable_grad() if requires_grad else torch.no_grad()
    with context:
        with _FixedEditHook(runtime.layer_module, edit):
            with _SupportOnlyOutputHead(
                runtime.output_embeddings,
                support_ids,
                prompt_len=prompt_len,
                response_len=response_len,
                chunk_size=chunk_size,
            ):
                logits = runtime.model(input_ids=input_ids, use_cache=False).logits
    return logits[:, prompt_len - 1 : prompt_len - 1 + response_len, :]


def _response_logits(
    runtime: Runtime,
    input_ids: torch.Tensor,
    *,
    prompt_len: int,
    response_len: int,
    edit: torch.Tensor | None,
    requires_grad: bool = False,
) -> torch.Tensor:
    context = torch.enable_grad() if requires_grad else torch.no_grad()
    with context:
        if edit is None:
            logits = runtime.model(input_ids=input_ids, use_cache=False).logits
        else:
            with _FixedEditHook(runtime.layer_module, edit):
                logits = runtime.model(input_ids=input_ids, use_cache=False).logits
    return logits[:, prompt_len - 1 : prompt_len - 1 + response_len, :]


def _conditional_probabilities(
    logp: torch.Tensor, support_mask: torch.Tensor
) -> torch.Tensor:
    mask = support_mask.to(device=logp.device, dtype=torch.bool)
    masked = logp.float().masked_fill(~mask, float("-inf"))
    normalized = masked - torch.logsumexp(masked, dim=-1, keepdim=True)
    return normalized.exp().masked_fill(~mask, 0.0)


def _top_support_probability_changes(
    tokenizer,
    support_ids: torch.Tensor,
    support_mask: torch.Tensor,
    source_logp: torch.Tensor,
    target_logp: torch.Tensor,
    edited_logp: torch.Tensor,
    *,
    top_n: int = 5,
) -> list[list[dict]]:
    """Summarize largest conditional-support shifts without Python sorting."""
    mask = support_mask.detach().bool().cpu()
    source = _conditional_probabilities(source_logp.detach().cpu(), mask)
    target = _conditional_probabilities(target_logp.detach().cpu(), mask)
    edited = _conditional_probabilities(edited_logp.detach().cpu(), mask)
    if int(mask.shape[0]) == 0 or int(mask.shape[1]) == 0 or int(top_n) <= 0:
        return [[] for _ in range(int(mask.shape[0]))]
    magnitude = torch.maximum((target - source).abs(), (edited - source).abs())
    magnitude = magnitude.masked_fill(~mask, float("-inf"))
    k = min(int(top_n), int(mask.shape[1]))
    ranked = torch.topk(magnitude, k=k, dim=-1, largest=True, sorted=True).indices
    ranked_valid = mask.gather(-1, ranked)
    ranked_token_ids = support_ids.detach().long().cpu().gather(-1, ranked)
    decoded = token_texts(tokenizer, ranked_token_ids.reshape(-1))

    rows: list[list[dict]] = []
    flat_offset = 0
    for response_index in range(int(mask.shape[0])):
        token_rows = []
        for rank_index in range(k):
            token_text = decoded[flat_offset + rank_index]
            if not bool(ranked_valid[response_index, rank_index]):
                continue
            support_index = int(ranked[response_index, rank_index])
            token_id = int(ranked_token_ids[response_index, rank_index])
            token_rows.append(
                {
                    "token_id": token_id,
                    "token_text": token_text,
                    "source_probability": float(source[response_index, support_index]),
                    "target_probability": float(target[response_index, support_index]),
                    "edited_probability": float(edited[response_index, support_index]),
                    "target_minus_source": float(
                        target[response_index, support_index]
                        - source[response_index, support_index]
                    ),
                    "edited_minus_source": float(
                        edited[response_index, support_index]
                        - source[response_index, support_index]
                    ),
                    "edited_minus_target": float(
                        edited[response_index, support_index]
                        - target[response_index, support_index]
                    ),
                }
            )
        flat_offset += k
        rows.append(token_rows)
    return rows


def _record_id(source_id: str, response_id: str, direction: str, mode: str) -> str:
    raw = f"{mode}:{source_id}:{response_id}:{direction}"
    suffix = hashlib.sha256(raw.encode()).hexdigest()[:10]
    safe_source = "".join(char if char.isalnum() or char in "-_" else "_" for char in source_id)
    return f"{safe_source[:48]}_{direction}_{suffix}"


def validate_support_row(row: dict) -> None:
    response_ids = row["response_ids"]
    support_ids = row["support_ids"]
    support_mask = row["support_mask"]
    response_length = int(response_ids.numel())
    if (
        support_ids.ndim != 2
        or support_mask.shape != support_ids.shape
        or int(support_ids.shape[0]) != response_length
    ):
        raise RuntimeError(
            "Conditional support is not aligned with the response: "
            f"response={tuple(response_ids.shape)}, "
            f"support={tuple(support_ids.shape)}, "
            f"mask={tuple(support_mask.shape)}"
        )
    for key in ("logp_control", "logp_intervention"):
        value = row[key]
        if value.shape != support_ids.shape:
            raise RuntimeError(
                f"{key} shape {tuple(value.shape)} differs from support "
                f"shape {tuple(support_ids.shape)}"
            )
    actual_present = (
        (support_ids == response_ids[:, None]) & support_mask.bool()
    ).any(dim=-1)
    if not bool(actual_present.all()):
        missing = torch.where(~actual_present)[0][:10].tolist()
        raise RuntimeError(
            "Observed response token is missing from the conditional support "
            f"at response positions {missing}"
        )


def trace_record(
    runtime: Runtime,
    *,
    row: dict,
    direction: str,
    mode: str,
    response_id: str | None = None,
    generated_by: str = "",
    hidden_override: torch.Tensor | None = None,
    edit_override: torch.Tensor | None = None,
    edited_logits_override: torch.Tensor | None = None,
    edited_logp_override: torch.Tensor | None = None,
) -> tuple[dict, dict]:
    """Trace one fixed trajectory and its applied residual edits."""
    validate_support_row(row)
    prompt_ids = prompt_ids_for_direction(row, direction).detach().long().cpu()
    response_ids = row["response_ids"].detach().long().cpu()
    input_ids = torch.cat([prompt_ids, response_ids]).to(
        device=runtime.device, dtype=torch.long
    ).view(1, -1)
    prompt_len = int(prompt_ids.numel())
    response_len = int(response_ids.numel())
    trace = None
    if edit_override is not None:
        # The inline hook produces an edit in the model hidden dtype.  Preserve
        # it exactly for storage; later gradient-based stages cast to fp32.
        edit = edit_override.to(runtime.device)
    else:
        hidden = (
            hidden_override.to(runtime.device).float()
            if hidden_override is not None
            else capture_hidden(runtime, input_ids)
        )
        trace = trace_from_hidden(runtime.edit_module, hidden)
        edit = trace["edit_by_direction"][direction]
    support_ids = row["support_ids"].to(runtime.device, dtype=torch.long)
    support_mask = row["support_mask"].to(runtime.device, dtype=torch.bool)
    if edited_logp_override is not None:
        edited_logp = edited_logp_override.to(runtime.device).float()
        edited_support_mass = torch.full(
            (response_len,), float("nan"), device=runtime.device, dtype=torch.float32
        )
    else:
        edited_logits = (
            edited_logits_override.to(runtime.device).float()
            if edited_logits_override is not None
            else _response_logits(
                runtime,
                input_ids,
                prompt_len=prompt_len,
                response_len=response_len,
                edit=edit,
            )[0]
        )
        edited_logp, edited_support_mass = logp_on_support_from_logits(
            edited_logits, support_ids, support_mask
        )
    target_logp, target_other_logp = target_logps_for_direction(row, direction)
    source_logp, source_other_logp = source_logps_for_direction(row, direction)
    target_logp = target_logp.to(runtime.device).float()
    source_logp = source_logp.to(runtime.device).float()
    source_support_mass = (1.0 - source_other_logp.to(runtime.device).float().exp()).clamp(0.0, 1.0)
    target_support_mass = (1.0 - target_other_logp.to(runtime.device).float().exp()).clamp(0.0, 1.0)
    # This conservative coverage is exact from the teacher cache and remains
    # available when the edited forward projects only onto the fixed support.
    support_mass = torch.minimum(source_support_mass, target_support_mass)
    source_kl = conditional_topk_kl(target_logp, source_logp, support_mask)
    edited_kl = conditional_topk_kl(target_logp, edited_logp, support_mask)
    source_id = str(row.get("source_id", "unknown"))
    response_key = str(response_id or row.get("response_id", "response"))
    record_id = _record_id(source_id, response_key, direction, mode)
    token_ids = torch.cat([prompt_ids, response_ids])
    texts = token_texts(runtime.tokenizer, token_ids)
    token_rows = []
    for index, text in enumerate(texts):
        is_prompt = index < prompt_len
        response_index = index - prompt_len
        token_rows.append(
            {
                "token_index": index,
                "response_index": response_index if not is_prompt else None,
                "token_id": int(token_ids[index]),
                "token_text": text,
                "is_prompt": is_prompt,
            }
        )
    support_changes = _top_support_probability_changes(
        runtime.tokenizer,
        row["support_ids"],
        row["support_mask"],
        source_logp,
        target_logp,
        edited_logp,
    )
    source_mean = float(mean_token_kl(source_kl))
    edited_mean = float(mean_token_kl(edited_kl))
    summary = {
        "interpretation_coordinate_system": "applied_residual_edit",
        "record_id": record_id,
        "source_id": source_id,
        "response_id": response_key,
        "direction": direction,
        "mode": mode,
        "generated_by": generated_by,
        "prompt_tokens": prompt_len,
        "response_tokens": response_len,
        "finish_reason": row.get("finish_reason"),
        "response_truncated": bool(row.get("response_truncated", False)),
        "analysis_truncated_from_response_tokens": row.get(
            "analysis_truncated_from_response_tokens"
        ),
        "source_kl_mean": source_mean,
        "edited_kl_mean": edited_mean,
        "recovery": recovery_fraction(source_mean, edited_mean),
        "support_mass_p10": (
            float(torch.quantile(support_mass.float()[torch.isfinite(support_mass.float())], 0.1))
            if bool(torch.isfinite(support_mass.float()).any())
            else None
        ),
        "support_mass_min": (
            float(support_mass.float()[torch.isfinite(support_mass.float())].min())
            if bool(torch.isfinite(support_mass.float()).any())
            else None
        ),
        "support_mass_definition": "minimum_source_target_conditional_support_mass",
        "source_support_mass_p10": float(torch.quantile(source_support_mass, 0.1)),
        "source_support_mass_min": float(source_support_mass.min()),
        "target_support_mass_p10": float(torch.quantile(target_support_mass, 0.1)),
        "target_support_mass_min": float(target_support_mass.min()),
        "edited_support_mass_p10": (
            float(torch.quantile(edited_support_mass[torch.isfinite(edited_support_mass)], 0.1))
            if bool(torch.isfinite(edited_support_mass).any())
            else None
        ),
        "edited_support_mass_min": (
            float(edited_support_mass[torch.isfinite(edited_support_mass)].min())
            if bool(torch.isfinite(edited_support_mass).any())
            else None
        ),
        "source_prompt_text": decode_token_ids(runtime.tokenizer, prompt_ids.tolist()),
        "target_prompt_text": decode_token_ids(
            runtime.tokenizer, target_prompt_ids(row, direction).tolist()
        ),
        "tokens": token_rows,
        "support_changes": support_changes,
    }
    tensor_record = {
        **{
            key: value
            for key, value in summary.items()
            if key not in {"tokens", "support_changes"}
        },
        "prompt_ids": prompt_ids,
        "target_prompt_ids": target_prompt_ids(row, direction).detach().long().cpu(),
        "response_ids": response_ids,
        "support_ids": row["support_ids"].detach().long().cpu(),
        "support_mask": row["support_mask"].detach().bool().cpu(),
        "source_logp": source_logp.detach().cpu(),
        "target_logp": target_logp.detach().cpu(),
        "edited_logp": edited_logp.detach().cpu(),
        "source_kl": source_kl.detach().cpu(),
        "edited_kl": edited_kl.detach().cpu(),
        "support_mass": support_mass.detach().cpu(),
        "source_support_mass": source_support_mass.detach().cpu(),
        "target_support_mass": target_support_mass.detach().cpu(),
        "edited_support_mass": edited_support_mass.detach().cpu(),
        "edit": edit.detach().cpu(),
        "tokens": token_rows,
        "support_changes": support_changes,
    }
    return tensor_record, summary


def _summarize_records(records: list[dict], *, minimum_source_kl: float) -> dict:
    direction_rows = {}
    for direction in ("c_to_i", "i_to_c"):
        selected = [record for record in records if record["direction"] == direction]
        metrics = (
            aggregate_sequence_kl(
                [record["source_kl_mean"] for record in selected],
                [record["edited_kl_mean"] for record in selected],
            )
            if selected
            else {"source_kl_mean": 0.0, "edited_kl_mean": 0.0, "recovery": 0.0}
        )
        direction_rows[direction] = {"records": len(selected), **metrics}
    metrics = (
        aggregate_sequence_kl(
            [record["source_kl_mean"] for record in records],
            [record["edited_kl_mean"] for record in records],
        )
        if records
        else {"source_kl_mean": 0.0, "edited_kl_mean": 0.0, "recovery": 0.0}
    )
    threshold_source_means = []
    threshold_edited_means = []
    support = []
    source_support = []
    target_support = []
    edited_support = []
    for record in records:
        source_kl = record["source_kl"].float()
        edited_kl = record["edited_kl"].float()
        mask = source_kl > float(minimum_source_kl)
        if bool(mask.any()):
            threshold_source_means.append(float(source_kl[mask].mean()))
            threshold_edited_means.append(float(edited_kl[mask].mean()))
        support.append(record["support_mass"].float())
        source_support.append(record["source_support_mass"].float())
        target_support.append(record["target_support_mass"].float())
        edited_support.append(record.get("edited_support_mass", torch.empty(0)).float())
    support_values = torch.cat(support) if support else torch.empty(0)
    support_values = support_values[torch.isfinite(support_values)]
    source_support_values = torch.cat(source_support) if source_support else torch.empty(0)
    source_support_values = source_support_values[torch.isfinite(source_support_values)]
    target_support_values = torch.cat(target_support) if target_support else torch.empty(0)
    target_support_values = target_support_values[torch.isfinite(target_support_values)]
    edited_support_values = torch.cat(edited_support) if edited_support else torch.empty(0)
    edited_support_values = edited_support_values[torch.isfinite(edited_support_values)]
    return {
        "records": len(records),
        "source_prompts": len(
            {str(record.get("source_id", "")) for record in records}
        ),
        "response_tokens": sum(int(record["response_tokens"]) for record in records),
        **metrics,
        "minimum_source_kl": float(minimum_source_kl),
        "thresholded_recovery": recovery_fraction(
            float(np.mean(threshold_source_means)) if threshold_source_means else 0.0,
            float(np.mean(threshold_edited_means)) if threshold_edited_means else 0.0,
        ),
        "support_mass_p10": (
            float(torch.quantile(support_values, 0.1))
            if support_values.numel()
            else None
        ),
        "support_mass_min": (
            float(support_values.min()) if support_values.numel() else None
        ),
        "support_mass_definition": "minimum_source_target_conditional_support_mass",
        "source_support_mass_p10": (
            float(torch.quantile(source_support_values, 0.1))
            if source_support_values.numel()
            else None
        ),
        "source_support_mass_min": (
            float(source_support_values.min()) if source_support_values.numel() else None
        ),
        "target_support_mass_p10": (
            float(torch.quantile(target_support_values, 0.1))
            if target_support_values.numel()
            else None
        ),
        "target_support_mass_min": (
            float(target_support_values.min()) if target_support_values.numel() else None
        ),
        "edited_support_mass_p10": (
            float(torch.quantile(edited_support_values, 0.1))
            if edited_support_values.numel()
            else None
        ),
        "edited_support_mass_min": (
            float(edited_support_values.min())
            if edited_support_values.numel()
            else None
        ),
        "directions": direction_rows,
    }


def _write_support_mass_distribution(records: list[dict], output: Path) -> None:
    """Write exact tail statistics and histograms for retained support masses."""
    def collect(key: str) -> torch.Tensor:
        values = []
        for record in records:
            value = record[key]
            if value is not None:
                values.append(value.detach().float().cpu().reshape(-1))
        return torch.cat(values) if values else torch.empty(0)

    metric_values = {
        "minimum_source_target": collect("support_mass"),
        "source": collect("source_support_mass"),
        "target": collect("target_support_mass"),
        "edited": collect("edited_support_mass"),
    }
    summaries = {
        key: summarize_support_masses(values)
        for key, values in metric_values.items()
    }
    conservative = summaries["minimum_source_target"]
    write_json(
        output / "support_mass_distribution.json",
        {
            "population": "teacher_forced_response_positions_min_source_target",
            "support_mass_definition": "minimum_source_target_conditional_support_mass",
            "positions": conservative["histogram"]["positions"],
            "counts": conservative["histogram"]["counts"],
            "edges": conservative["histogram"]["edges"],
            "p10": conservative["p10"],
            "minimum": conservative["minimum"],
            "series": summaries,
        },
    )


def save_records(
    records: Iterable[tuple[dict, dict]],
    output: Path,
    *,
    minimum_source_kl: float,
) -> dict:
    record_dir = output / "records"
    record_dir.mkdir(parents=True, exist_ok=True)
    records = list(records)
    expected = {
        str(summary["record_id"])
        for _tensor_record, summary in records
    }
    for pattern in ("*.pt", "*.json"):
        for path in record_dir.glob(pattern):
            if path.stem not in expected:
                path.unlink()
    tensors = []
    summaries = []
    for tensor_record, summary in records:
        record_id = str(summary["record_id"])
        torch.save(tensor_record, record_dir / f"{record_id}.pt")
        write_json(record_dir / f"{record_id}.json", summary)
        tensors.append(tensor_record)
        summaries.append(summary)
    summary = _summarize_records(
        tensors, minimum_source_kl=float(minimum_source_kl)
    )
    summary["record_ids"] = [row["record_id"] for row in summaries]
    write_json(output / "summary.json", summary)
    _write_support_mass_distribution(tensors, output)
    return summary


def trace_teacher_forced(
    config: dict,
    runtime: Runtime,
    output: Path,
    *,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict:
    dataset = TopKKLDistillDataset(
        Path(config["data"]["cache_dir"]),
        str(config["data"]["evaluation_split"]),
    )
    rows = select_diverse_rows(dataset.rows, int(config["data"]["max_examples"]))
    shard_count = max(1, int(shard_count))
    shard_index = int(shard_index)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"invalid teacher-forced shard {shard_index}/{shard_count}")
    if shard_count > 1:
        # Balance deterministic teacher-forced work by estimated token cost, not
        # merely source count.  Both directions for a source stay on one worker.
        rows_by_source: dict[str, list[dict]] = defaultdict(list)
        for index, row in enumerate(rows):
            rows_by_source[str(row.get("source_id", index))].append(row)
        source_costs = []
        for source_id, source_rows in rows_by_source.items():
            cost = 0
            for row in source_rows:
                response_len = int(row["response_ids"].numel())
                for direction in (
                    ["c_to_i", "i_to_c"]
                    if config["edit"]["directions"] == "both"
                    else [str(config["edit"]["directions"])]
                ):
                    cost += int(prompt_ids_for_direction(row, direction).numel()) + response_len
            source_costs.append((int(cost), source_id))
        shard_loads = [0 for _ in range(shard_count)]
        source_to_shard = {}
        for cost, source_id in sorted(source_costs, key=lambda item: (-item[0], item[1])):
            destination = min(range(shard_count), key=lambda index: (shard_loads[index], index))
            source_to_shard[source_id] = destination
            shard_loads[destination] += int(cost)
        rows = [
            row
            for index, row in enumerate(rows)
            if source_to_shard[str(row.get("source_id", index))] == shard_index
        ]
    directions = (
        ["c_to_i", "i_to_c"]
        if config["edit"]["directions"] == "both"
        else [str(config["edit"]["directions"])]
    )
    records = []
    response_cap = int(config["data"].get("analysis_response_token_cap", 0))
    jobs = [
        {
            "row": truncate_cache_row(row, response_cap),
            "direction": direction,
            "mode": "teacher_forced",
            "generated_by": "cached_fixed_prefix",
        }
        for row in rows
        for direction in directions
    ]
    # Batching variable-length fixed prefixes without bucketing can be slower
    # than single-example execution because every sequence pays for the longest
    # member of its batch. Keep similarly sized trajectories together.
    jobs.sort(
        key=lambda job: int(
            prompt_ids_for_direction(job["row"], job["direction"]).numel()
            + job["row"]["response_ids"].numel()
        )
    )
    # ``batch_size`` is a maximum.  Because jobs are sorted from short to long,
    # start aggressively and permanently back off only when a later length
    # bucket no longer fits.  This uses large-memory GPUs efficiently without
    # paying for the same failed oversized batch on every subsequent bucket.
    max_batch_size = max(1, int(config["teacher_forced"].get("batch_size", 1)))
    active_batch_size = max_batch_size
    progress = tqdm(
        total=len(jobs),
        desc=f"teacher-forced shard {shard_index + 1}/{shard_count}",
        unit="record",
    )
    cursor = 0
    while cursor < len(jobs):
        batch = jobs[cursor : cursor + active_batch_size]
        try:
            records.extend(trace_records_batch(runtime, batch))
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or active_batch_size <= 1:
                raise
            previous = active_batch_size
            active_batch_size = max(1, active_batch_size // 2)
            if torch.cuda.is_available() and str(runtime.device).startswith("cuda"):
                torch.cuda.empty_cache()
            progress.write(
                f"teacher-forced shard {shard_index + 1}/{shard_count}: "
                f"CUDA OOM at batch={previous}; continuing with batch={active_batch_size}"
            )
            continue
        cursor += len(batch)
        progress.update(len(batch))
    progress.close()
    return save_records(
        records,
        output,
        minimum_source_kl=float(
            config["teacher_forced"].get("min_source_kl", 0.0)
        ),
    )


def merge_teacher_forced_shards(
    shard_directories: list[Path],
    output: Path,
    *,
    minimum_source_kl: float,
) -> dict:
    """Merge disjoint source-sharded teacher-forced records."""
    output.mkdir(parents=True, exist_ok=True)
    records_output = output / "records"
    records_output.mkdir(parents=True, exist_ok=True)
    for path in records_output.glob("*"):
        if path.is_file():
            path.unlink()

    seen: set[str] = set()
    for directory in shard_directories:
        record_dir = directory / "records"
        if not record_dir.exists():
            continue
        for source in sorted(record_dir.glob("*")):
            if not source.is_file():
                continue
            key = source.stem
            if key in seen and source.suffix == ".pt":
                raise RuntimeError(f"Duplicate teacher-forced shard record: {key}")
            if source.suffix == ".pt":
                seen.add(key)
            source.replace(records_output / source.name)

    records = load_records(output)
    summary = _summarize_records(
        records, minimum_source_kl=float(minimum_source_kl)
    )
    summary["record_ids"] = [str(record["record_id"]) for record in records]
    write_json(output / "summary.json", summary)
    _write_support_mass_distribution(records, output)
    return summary


def _right_pad_sequences(
    sequences: list[torch.Tensor],
    *,
    pad_token_id: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(int(sequence.numel()) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), width),
        int(pad_token_id),
        device=device,
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, sequence in enumerate(sequences):
        length = int(sequence.numel())
        input_ids[index, :length] = sequence.to(device=device, dtype=torch.long)
        attention_mask[index, :length] = 1
    return input_ids, attention_mask


@torch.inference_mode()
def trace_records_batch(
    runtime: Runtime,
    specifications: list[dict],
) -> list[tuple[dict, dict]]:
    """Trace variable-length fixed sequences using sparse support projection."""
    if not specifications:
        return []
    sequences = []
    prompt_lengths = []
    response_lengths = []
    support_ids_batch = []
    for specification in specifications:
        row = specification["row"]
        prompt = prompt_ids_for_direction(
            row, specification["direction"]
        ).detach().long().cpu()
        response = row["response_ids"].detach().long().cpu()
        sequences.append(torch.cat([prompt, response]))
        prompt_lengths.append(int(prompt.numel()))
        response_lengths.append(int(response.numel()))
        support_ids_batch.append(row["support_ids"].detach().long().cpu())
    input_ids, attention_mask = _right_pad_sequences(
        sequences,
        pad_token_id=int(runtime.tokenizer.pad_token_id),
        device=runtime.device,
    )
    direction_values = torch.tensor(
        [direction_sign(str(specification["direction"])) for specification in specifications],
        dtype=torch.float32,
    )
    with _InlineDirectionalEditHook(
        runtime.layer_module, runtime.edit_module, direction_values
    ) as hook:
        with _BatchedSupportOnlyOutputHead(
            runtime.output_embeddings,
            support_ids_batch,
            prompt_lengths=prompt_lengths,
            response_lengths=response_lengths,
            chunk_positions=2048,
        ):
            support_logits_all = runtime.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
    if hook.edit is None:
        raise RuntimeError("inline edit hook did not observe the edit layer")
    edit = hook.edit
    records = []
    for index, specification in enumerate(specifications):
        valid = int(sequences[index].numel())
        prompt_len = prompt_lengths[index]
        response_len = response_lengths[index]
        width = int(specification["row"]["support_ids"].shape[1])
        support_logits = support_logits_all[
            index,
            :response_len,
            :width,
        ]
        support_mask = specification["row"]["support_mask"].to(
            runtime.device, dtype=torch.bool
        )
        edited_logp = conditional_logp_from_support_logits(
            support_logits, support_mask
        )
        records.append(
            trace_record(
                runtime,
                row=specification["row"],
                direction=str(specification["direction"]),
                mode=str(specification["mode"]),
                response_id=specification.get("response_id"),
                generated_by=str(specification.get("generated_by", "")),
                edit_override=edit[index : index + 1, :valid],
                edited_logp_override=edited_logp,
            )
        )
    return records


def trace_records_adaptive(
    runtime: Runtime,
    specifications: list[dict],
) -> list[tuple[dict, dict]]:
    """Trace a large batch and split only if CUDA memory is actually exceeded."""
    if not specifications:
        return []
    try:
        return trace_records_batch(runtime, specifications)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" not in message or len(specifications) <= 1:
            raise
        if torch.cuda.is_available() and str(runtime.device).startswith("cuda"):
            torch.cuda.empty_cache()
        middle = len(specifications) // 2
        return trace_records_adaptive(
            runtime, specifications[:middle]
        ) + trace_records_adaptive(runtime, specifications[middle:])


def _record_paths(
    directory: Path,
    *,
    maximum: int = 0,
    shard_index: int = 0,
    shard_count: int = 1,
    record_ids: set[str] | None = None,
) -> list[Path]:
    """Select record files before deserializing them.

    This matters for multi-GPU attribution: each worker should only unpickle its
    own shard rather than loading the complete dataset and slicing afterward.
    """
    paths = sorted((directory / "records").glob("*.pt"))
    if record_ids is not None:
        wanted = {str(value) for value in record_ids}
        paths = [path for path in paths if path.stem in wanted]
    if int(maximum) > 0:
        grouped: dict[str, dict[str, list[Path]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for path in paths:
            metadata = read_json(path.with_suffix(".json"), {})
            direction = str(metadata.get("direction", "unspecified"))
            source_id = str(metadata.get("source_id", path.stem))
            grouped[direction][source_id].append(path)
        selected: list[Path] = []
        for direction in sorted(grouped):
            direction_selected: list[Path] = []
            depth = 0
            while len(direction_selected) < int(maximum):
                added = False
                for source_id in sorted(grouped[direction]):
                    candidates = grouped[direction][source_id]
                    if depth < len(candidates):
                        direction_selected.append(candidates[depth])
                        added = True
                        if len(direction_selected) >= int(maximum):
                            break
                if not added:
                    break
                depth += 1
            selected.extend(direction_selected)
        paths = selected
    shard_count = max(1, int(shard_count))
    shard_index = int(shard_index)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"invalid record shard {shard_index}/{shard_count}")
    if shard_count > 1:
        # Attribution/ablation cost scales much more closely with sequence
        # length than record count.  Greedily balance the lightweight JSON
        # sidecar token counts so the final long-record tail is shared across
        # GPUs instead of leaving one worker running after the others finish.
        weighted = []
        for path in paths:
            metadata = read_json(path.with_suffix(".json"), {})
            cost = max(
                1,
                int(metadata.get("prompt_tokens", 0))
                + int(metadata.get("response_tokens", 0)),
            )
            weighted.append((cost, path))
        shard_paths: list[list[Path]] = [[] for _ in range(shard_count)]
        shard_loads = [0 for _ in range(shard_count)]
        for cost, path in sorted(weighted, key=lambda item: (-item[0], item[1].name)):
            destination = min(range(shard_count), key=lambda index: (shard_loads[index], index))
            shard_paths[destination].append(path)
            shard_loads[destination] += int(cost)
        paths = sorted(shard_paths[shard_index])
    return paths


def iter_records(
    directory: Path,
    *,
    maximum: int = 0,
    shard_index: int = 0,
    shard_count: int = 1,
    record_ids: set[str] | None = None,
):
    """Yield records one at a time to bound peak host memory."""
    for path in _record_paths(
        directory,
        maximum=maximum,
        shard_index=shard_index,
        shard_count=shard_count,
        record_ids=record_ids,
    ):
        yield torch.load(path, map_location="cpu")


def load_records(
    directory: Path,
    *,
    maximum: int = 0,
    shard_index: int = 0,
    shard_count: int = 1,
    record_ids: set[str] | None = None,
) -> list[dict]:
    return list(
        iter_records(
            directory,
            maximum=maximum,
            shard_index=shard_index,
            shard_count=shard_count,
            record_ids=record_ids,
        )
    )


def load_record_summaries(directory: Path) -> list[dict]:
    """Load lightweight JSON sidecars instead of tensor-heavy record files.

    If any JSON sidecars exist, return those summaries in filename order.
    Otherwise load the tensor records. A directory with some sidecars present
    does not fall back individually for records whose sidecars are missing.
    """
    record_dir = directory / "records"
    json_paths = sorted(record_dir.glob("*.json"))
    if json_paths:
        return [read_json(path, {}) for path in json_paths]
    return [torch.load(path, map_location="cpu") for path in sorted(record_dir.glob("*.pt"))]
