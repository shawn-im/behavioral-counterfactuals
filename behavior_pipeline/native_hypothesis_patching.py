from __future__ import annotations

"""Internal patching implementation for :mod:`behavior_pipeline.native_hypothesis`."""


import hashlib
import json
import random
import shutil
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .counterfactual_runtime import Runtime, _BatchedResponseOnlyOutputHead, _right_pad_sequences, write_json, qualitative_record_key
from .token_text import token_id_texts
from .topk_kl import conditional_topk_kl, logp_on_support_from_logits, recovery_fraction
from .native_hypothesis_sites import _display_token, _site_metric_aggregate, canonical_direction_sign, detect_patch_sites


class _ActivationReplaceHook:
    """Replace selected layer-output positions with target-condition activations."""

    def __init__(
        self,
        layer_module: nn.Module,
        source_indices: list[int],
        target_values: torch.Tensor,
    ):
        self.layer_module = layer_module
        self.source_indices = [int(value) for value in source_indices]
        self.target_values = target_values
        self.handle = None

    def _hook(self, _module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        is_list = isinstance(output, list)
        hidden = output[0] if (is_tuple or is_list) else output
        if not self.source_indices:
            return output
        if int(self.target_values.shape[1]) != len(self.source_indices):
            raise RuntimeError("Patch target count does not match source patch positions")
        patched = hidden.clone()
        indices = torch.tensor(self.source_indices, device=hidden.device, dtype=torch.long)
        patched.index_copy_(
            1,
            indices,
            self.target_values.to(device=hidden.device, dtype=hidden.dtype),
        )
        if is_tuple:
            return (patched, *output[1:])
        if is_list:
            values = list(output)
            values[0] = patched
            return values
        return patched

    def __enter__(self):
        self.handle = self.layer_module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class _BatchedActivationReplaceHook:
    """Apply a different activation replacement to each batch row."""

    def __init__(
        self,
        layer_module: nn.Module,
        source_indices: list[list[int]],
        target_values: list[torch.Tensor],
    ):
        if len(source_indices) != len(target_values):
            raise ValueError("batched patch indices/values must have equal length")
        self.layer_module = layer_module
        self.source_indices = [[int(value) for value in row] for row in source_indices]
        self.target_values = target_values
        self.handle = None

    def _hook(self, _module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        is_list = isinstance(output, list)
        hidden = output[0] if (is_tuple or is_list) else output
        if int(hidden.shape[0]) != len(self.source_indices):
            raise RuntimeError("patch batch size does not match model batch")
        patched = hidden.clone()
        for batch_index, (indices, values) in enumerate(
            zip(self.source_indices, self.target_values)
        ):
            if not indices:
                continue
            if int(values.shape[1]) != len(indices):
                raise RuntimeError("Patch target count does not match source patch positions")
            index_tensor = torch.tensor(indices, device=hidden.device, dtype=torch.long)
            patched[batch_index].index_copy_(
                0,
                index_tensor,
                values[0].to(device=hidden.device, dtype=hidden.dtype),
            )
        if is_tuple:
            return (patched, *output[1:])
        if is_list:
            values = list(output)
            values[0] = patched
            return values
        return patched

    def __enter__(self):
        self.handle = self.layer_module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class _HiddenCapture:
    def __init__(self, layer_module: nn.Module):
        self.layer_module = layer_module
        self.hidden = None
        self.handle = None

    def _hook(self, _module, _inputs, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        self.hidden = hidden.detach()

    def __enter__(self):
        self.handle = self.layer_module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def _response_slice(logits: torch.Tensor, *, prompt_len: int, response_len: int) -> torch.Tensor:
    first = int(prompt_len) - 1
    last = first + int(response_len)
    return logits[:, first:last, :]


def _summarize_response_logits(
    logits: torch.Tensor,
    *,
    support_ids: torch.Tensor,
    support_mask: torch.Tensor,
    response_ids: torch.Tensor,
    batch_index: int = 0,
) -> dict[str, torch.Tensor]:
    """Reduce full-vocabulary logits immediately to the quantities we retain."""
    logits = logits[int(batch_index)]
    support_logp, support_mass = logp_on_support_from_logits(
        logits,
        support_ids.to(logits.device),
        support_mask.to(logits.device),
    )
    support_prob = support_logp.exp().masked_fill(
        ~support_mask.to(logits.device, dtype=torch.bool), 0.0
    )
    observed_ids = response_ids.detach().long().to(logits.device)[: logits.shape[0]]
    observed_logits = logits.gather(-1, observed_ids.view(-1, 1)).squeeze(-1).float()
    observed_logp = observed_logits - torch.logsumexp(logits.float(), dim=-1)
    return {
        "support_logp": support_logp.detach().float().cpu(),
        "support_probability": support_prob.detach().float().cpu(),
        "support_mass": support_mass.detach().float().cpu(),
        "observed_token_logp": observed_logp.detach().float().cpu(),
        "observed_token_probability": observed_logp.exp().detach().float().cpu(),
    }


def _native_response_head_context(
    runtime: Runtime, *, prompt_len: int, response_len: int
):
    """Project native-analysis logits only at response-prediction positions."""
    if isinstance(runtime.output_embeddings, nn.Linear):
        return _BatchedResponseOnlyOutputHead(
            runtime.output_embeddings,
            prompt_lengths=[int(prompt_len)],
            response_lengths=[int(response_len)],
        )
    return None


def _native_response_head_batch_context(
    runtime: Runtime,
    *,
    prompt_lengths: list[int],
    response_lengths: list[int],
):
    if isinstance(runtime.output_embeddings, nn.Linear):
        return _BatchedResponseOnlyOutputHead(
            runtime.output_embeddings,
            prompt_lengths=[int(value) for value in prompt_lengths],
            response_lengths=[int(value) for value in response_lengths],
        )
    return None


def _pad_token_id(tokenizer) -> int:
    value = getattr(tokenizer, "pad_token_id", None)
    if value is None:
        value = getattr(tokenizer, "eos_token_id", None)
    return int(value if value is not None else 0)


@torch.no_grad()
def _forward_patch_conditions(
    runtime: Runtime,
    source_sequence: torch.Tensor,
    *,
    prompt_len: int,
    response_len: int,
    support_ids: torch.Tensor,
    support_mask: torch.Tensor,
    response_ids: torch.Tensor,
    source_indices: list[list[int]],
    target_values: list[torch.Tensor],
) -> list[dict[str, torch.Tensor]]:
    """Evaluate native patch conditions together in one transformer batch."""
    count = len(source_indices)
    if count == 0:
        return []
    sequence = source_sequence.detach().long().to(runtime.device)
    input_ids = sequence.view(1, -1).expand(count, -1)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    head_context = _native_response_head_batch_context(
        runtime,
        prompt_lengths=[int(prompt_len)] * count,
        response_lengths=[int(response_len)] * count,
    )
    with _BatchedActivationReplaceHook(
        runtime.layer_module, source_indices, target_values
    ):
        if head_context is None:
            result = runtime.model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            )
            compact = _response_slice(
                result.logits, prompt_len=prompt_len, response_len=response_len
            )
        else:
            with head_context:
                result = runtime.model(
                    input_ids=input_ids, attention_mask=attention_mask, use_cache=False
                )
            compact = result.logits[:, :response_len, :]
    summaries = [
        _summarize_response_logits(
            compact,
            support_ids=support_ids,
            support_mask=support_mask,
            response_ids=response_ids,
            batch_index=index,
        )
        for index in range(count)
    ]
    del result, compact, input_ids, attention_mask
    return summaries


@torch.no_grad()
def _forward_source_or_target(
    runtime: Runtime,
    input_ids: torch.Tensor,
    *,
    prompt_len: int,
    response_len: int,
    support_ids: torch.Tensor,
    support_mask: torch.Tensor,
    response_ids: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    head_context = _native_response_head_context(
        runtime, prompt_len=prompt_len, response_len=response_len
    )
    with _HiddenCapture(runtime.layer_module) as capture:
        if head_context is None:
            result = runtime.model(input_ids=input_ids, use_cache=False)
            response_logits = _response_slice(
                result.logits, prompt_len=prompt_len, response_len=response_len
            )
        else:
            with head_context:
                result = runtime.model(input_ids=input_ids, use_cache=False)
            response_logits = result.logits[:, :response_len, :]
    if capture.hidden is None:
        raise RuntimeError("Native analysis failed to capture the patch-layer residual")
    summary = _summarize_response_logits(
        response_logits,
        support_ids=support_ids,
        support_mask=support_mask,
        response_ids=response_ids,
    )
    del result, response_logits
    return capture.hidden.detach(), summary


@torch.no_grad()
def _forward_with_patch(
    runtime: Runtime,
    input_ids: torch.Tensor,
    *,
    prompt_len: int,
    response_len: int,
    support_ids: torch.Tensor,
    support_mask: torch.Tensor,
    response_ids: torch.Tensor,
    source_indices: list[int],
    target_values: torch.Tensor,
) -> dict[str, torch.Tensor]:
    head_context = _native_response_head_context(
        runtime, prompt_len=prompt_len, response_len=response_len
    )
    with _ActivationReplaceHook(
        runtime.layer_module, source_indices, target_values
    ):
        if head_context is None:
            result = runtime.model(input_ids=input_ids, use_cache=False)
            response_logits = _response_slice(
                result.logits, prompt_len=prompt_len, response_len=response_len
            )
        else:
            with head_context:
                result = runtime.model(input_ids=input_ids, use_cache=False)
            response_logits = result.logits[:, :response_len, :]
    summary = _summarize_response_logits(
        response_logits,
        support_ids=support_ids,
        support_mask=support_mask,
        response_ids=response_ids,
    )
    del result, response_logits
    return summary


def _aligned_source_target_pairs(
    source_prompt_ids: torch.Tensor,
    target_prompt_ids: torch.Tensor,
    *,
    response_len: int,
) -> list[tuple[int, int]]:
    """Return deterministic token-aligned source↔target positions.

    Prompt positions are drawn only from exact equal-token blocks under a
    sequence alignment; response positions are aligned directly because the
    fixed response is identical under source and target conditions.
    """
    source_prompt = [int(value) for value in source_prompt_ids.tolist()]
    target_prompt = [int(value) for value in target_prompt_ids.tolist()]
    pairs: list[tuple[int, int]] = []
    matcher = SequenceMatcher(a=source_prompt, b=target_prompt, autojunk=False)
    for block in matcher.get_matching_blocks():
        for offset in range(int(block.size)):
            pairs.append((int(block.a + offset), int(block.b + offset)))
    source_offset = len(source_prompt)
    target_offset = len(target_prompt)
    pairs.extend(
        (source_offset + index, target_offset + index)
        for index in range(int(response_len))
    )
    # Preserve source order while removing duplicate source positions.
    unique: dict[int, int] = {}
    for source_index, target_index in pairs:
        unique.setdefault(int(source_index), int(target_index))
    return sorted(unique.items())


def _combine_patch_sites(sites: dict, kinds: list[str]) -> dict:
    mapping: dict[int, int] = {}
    for kind in kinds:
        site = sites.get(kind, {})
        source = [int(value) for value in site.get("source_indices", [])]
        target = [int(value) for value in site.get("target_indices", [])]
        if len(source) != len(target):
            continue
        for source_index, target_index in zip(source, target):
            previous = mapping.get(source_index)
            if previous is not None and previous != target_index:
                raise RuntimeError(
                    "Combined native patch maps one source position to multiple target positions"
                )
            mapping[source_index] = target_index
    ordered = sorted(mapping.items())
    return {
        "source_indices": [row[0] for row in ordered],
        "target_indices": [row[1] for row in ordered],
    }


def _random_patch_control(
    *,
    aligned_pairs: list[tuple[int, int]],
    excluded_source_indices: set[int],
    count: int,
    seed: int,
) -> dict:
    """Select a deterministic random aligned-token control patch of equal size."""
    candidates = [
        (int(source), int(target))
        for source, target in aligned_pairs
        if int(source) not in excluded_source_indices
    ]
    if int(count) <= 0 or len(candidates) < int(count):
        return {"source_indices": [], "target_indices": []}
    rng = random.Random(int(seed))
    chosen = rng.sample(candidates, int(count))
    chosen.sort(key=lambda row: row[0])
    return {
        "source_indices": [row[0] for row in chosen],
        "target_indices": [row[1] for row in chosen],
    }


def _conditional_kl_metrics(record: dict, summary: dict[str, torch.Tensor]) -> dict:
    """Score a native condition against the cached counterfactual target on fixed support."""
    support_mask = record["support_mask"].detach().bool().cpu()
    target_logp = record["target_logp"].detach().float().cpu()
    source_kl = record["source_kl"].detach().float().cpu()
    condition_kl = conditional_topk_kl(
        target_logp, summary["support_logp"].float(), support_mask
    ).detach().float().cpu()
    source_mean = float(source_kl.mean())
    condition_mean = float(condition_kl.mean())
    return {
        "source_kl_mean": source_mean,
        "condition_kl_mean": condition_mean,
        "recovery": recovery_fraction(source_mean, condition_mean),
        "per_response_kl": [float(value) for value in condition_kl.tolist()],
        "per_response_kl_recovered": [
            float(value) for value in (source_kl - condition_kl).tolist()
        ],
        "per_response_observed_token_probability": [
            float(value) for value in summary["observed_token_probability"].tolist()
        ],
        "per_response_observed_token_logp": [
            float(value) for value in summary["observed_token_logp"].tolist()
        ],
    }


def _native_qualitative_probability_table(
    tokenizer,
    support_ids: torch.Tensor,
    support_mask: torch.Tensor,
    *,
    source_summary: dict[str, torch.Tensor],
    target_summary: dict[str, torch.Tensor],
    patch_summary: dict[str, torch.Tensor],
    top_n: int = 20,
) -> list[dict]:
    """Compact Source/Target/Native-patch probabilities for HTML example callouts."""
    support_ids = support_ids.detach().long().cpu()
    support_mask = support_mask.detach().bool().cpu()
    conditions = {
        "source": source_summary["support_probability"].detach().float().cpu(),
        "target": target_summary["support_probability"].detach().float().cpu(),
        "native_patch": patch_summary["support_probability"].detach().float().cpu(),
    }
    rows: list[dict] = []
    for response_index in range(int(support_ids.shape[0])):
        valid = torch.nonzero(support_mask[response_index], as_tuple=False).flatten()
        if valid.numel() == 0:
            rows.append({"response_index": response_index, "alternatives": []})
            continue
        matrix = torch.stack(
            [conditions[name][response_index, valid] for name in conditions], dim=0
        )
        separation = (conditions["native_patch"][response_index, valid]
                      - conditions["source"][response_index, valid]).abs()
        max_probability = matrix.max(dim=0).values
        ranked = sorted(
            range(int(valid.numel())),
            key=lambda local: (
                -float(separation[local]),
                -float(max_probability[local]),
                int(valid[local]),
            ),
        )[: max(3, int(top_n))]
        alternatives = []
        for local in ranked:
            support_col = int(valid[local])
            token_id = int(support_ids[response_index, support_col])
            token_text = token_id_texts(tokenizer, [token_id])[0]
            alternatives.append({
                "token_id": token_id,
                "token_text": str(token_text),
                **{
                    name: float(values[response_index, support_col])
                    for name, values in conditions.items()
                },
            })
        rows.append({"response_index": response_index, "alternatives": alternatives})
    return rows


def _context_for_position(record: dict, response_index: int, *, radius: int = 10) -> str:
    tokens = list(record.get("tokens", []))
    prompt_len = int(record.get("prompt_tokens", 0))
    global_index = prompt_len + int(response_index)
    start = max(prompt_len, global_index - int(radius))
    end = min(len(tokens), global_index + 4)
    parts = []
    for index in range(start, end):
        text = _display_token(str(tokens[index].get("token_text", "")))
        if index == global_index:
            parts.append(f"⟦{text}⟧")
        else:
            parts.append(text)
    return "".join(parts).replace("\n", "↵")[:500]


def _shift_details(
    record: dict,
    *,
    support_ids: torch.Tensor,
    support_mask: torch.Tensor,
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    canonical_sign: float,
    tokenizer,
    top_tokens_per_position: int,
    top_positions: int,
) -> dict:
    """Return literal probability movements for qualitative inspection."""
    ids = support_ids.detach().long().cpu()
    mask = support_mask.detach().bool().cpu()
    before_prob = before["support_probability"].float()
    after_prob = after["support_probability"].float()
    delta = (after_prob - before_prob) * float(canonical_sign)
    before_other = (1.0 - before["support_mass"].float()).clamp(0.0, 1.0)
    after_other = (1.0 - after["support_mass"].float()).clamp(0.0, 1.0)
    other_delta = (after_other - before_other) * float(canonical_sign)
    moved = []
    position_payload = []
    vocabulary_text: dict[int, str] = {}
    for response_index in range(int(ids.shape[0])):
        valid = torch.where(mask[response_index])[0]
        values = delta[response_index, valid]
        token_ids = ids[response_index, valid]
        partition_tv = 0.5 * (
            float(values.abs().sum()) + abs(float(other_delta[response_index]))
        )
        moved.append(float(partition_tv))
        order_positive = torch.argsort(values, descending=True).tolist()
        order_negative = torch.argsort(values, descending=False).tolist()

        def rows(order, positive: bool):
            result = []
            for local in order:
                value = float(values[local])
                if (positive and value <= 0.0) or ((not positive) and value >= 0.0):
                    continue
                token_id = int(token_ids[local])
                if token_id not in vocabulary_text:
                    vocabulary_text[token_id] = runtime_decode_token(tokenizer, token_id)
                support_index = int(valid[local])
                result.append(
                    {
                        "token_id": token_id,
                        "token_text": vocabulary_text[token_id],
                        "canonical_delta_probability": value,
                        "before_probability": float(before_prob[response_index, support_index]),
                        "after_probability": float(after_prob[response_index, support_index]),
                    }
                )
                if len(result) >= int(top_tokens_per_position):
                    break
            return result

        position_payload.append(
            {
                "response_index": int(response_index),
                "observed_token": _display_token(
                    str(record["tokens"][int(record["prompt_tokens"]) + response_index].get("token_text", ""))
                ),
                "moved_mass": float(partition_tv),
                "other_delta_probability": float(other_delta[response_index]),
                "top_gains": rows(order_positive, True),
                "top_losses": rows(order_negative, False),
                "context": _context_for_position(record, response_index),
            }
        )
    ranked_positions = sorted(
        position_payload, key=lambda row: float(row["moved_mass"]), reverse=True
    )[: int(top_positions)]
    return {
        "mean_moved_mass": float(np.mean(moved)) if moved else 0.0,
        "max_moved_mass": float(max(moved)) if moved else 0.0,
        "mean_before_support_mass": float(before["support_mass"].float().mean()),
        "mean_after_support_mass": float(after["support_mass"].float().mean()),
        "mean_canonical_other_delta_probability": float(other_delta.float().mean()),
        "moved_mass_by_response_position": moved,
        "top_positions": ranked_positions,
    }


def runtime_decode_token(tokenizer, token_id: int) -> str:
    try:
        return token_id_texts(tokenizer, [int(token_id)])[0]
    except Exception:
        return str(int(token_id))


def _cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _forward_patch_spec_chunk_with_oom_fallback(
    runtime: Runtime,
    source_sequence: torch.Tensor,
    source_input: torch.Tensor,
    *,
    prompt_len: int,
    response_len: int,
    support_ids: torch.Tensor,
    support_mask: torch.Tensor,
    response_ids: torch.Tensor,
    chunk: list[dict],
) -> list[dict[str, torch.Tensor]]:
    """Evaluate a patch-condition chunk, recursively halving on CUDA OOM."""
    if not chunk:
        return []
    if len(chunk) == 1:
        row = chunk[0]
        return [_forward_with_patch(
            runtime,
            source_input,
            prompt_len=prompt_len,
            response_len=response_len,
            support_ids=support_ids,
            support_mask=support_mask,
            response_ids=response_ids,
            source_indices=row["source_indices"],
            target_values=row["target_values"],
        )]
    try:
        return _forward_patch_conditions(
            runtime,
            source_sequence,
            prompt_len=prompt_len,
            response_len=response_len,
            support_ids=support_ids,
            support_mask=support_mask,
            response_ids=response_ids,
            source_indices=[row["source_indices"] for row in chunk],
            target_values=[row["target_values"] for row in chunk],
        )
    except RuntimeError as exc:
        if not _cuda_oom(exc):
            raise
        if torch.cuda.is_available() and str(runtime.device).startswith("cuda"):
            torch.cuda.empty_cache()
        smaller = max(1, len(chunk) // 2)
        print(
            "native activation patching: CUDA OOM at condition_batch="
            f"{len(chunk)}; continuing with condition_batch<={smaller}",
            flush=True,
        )
        return (
            _forward_patch_spec_chunk_with_oom_fallback(
                runtime, source_sequence, source_input,
                prompt_len=prompt_len, response_len=response_len,
                support_ids=support_ids, support_mask=support_mask,
                response_ids=response_ids,
                chunk=chunk[:smaller],
            )
            + _forward_patch_spec_chunk_with_oom_fallback(
                runtime, source_sequence, source_input,
                prompt_len=prompt_len, response_len=response_len,
                support_ids=support_ids, support_mask=support_mask,
                response_ids=response_ids,
                chunk=chunk[smaller:],
            )
        )


def _prepare_native_record(config: dict, runtime: Runtime, record: dict) -> dict:
    """Prepare CPU-side native-analysis metadata without running the model."""
    settings = config.get("native_analysis", {})
    source_prompt_ids = record["prompt_ids"].detach().long().cpu()
    target_prompt_ids = record["target_prompt_ids"].detach().long().cpu()
    response_ids = record["response_ids"].detach().long().cpu()
    source_sequence = torch.cat([source_prompt_ids, response_ids])
    target_sequence = torch.cat([target_prompt_ids, response_ids])
    support_ids = record["support_ids"].detach().long().cpu()
    support_mask = record["support_mask"].detach().bool().cpu()
    response_len = int(response_ids.numel())

    sites = detect_patch_sites(
        record,
        runtime.tokenizer,
    )
    patch_kinds = [str(value) for value in settings.get(
        "patch_sites",
        ["filler", "question_span", "question_and_filler"],
    )]
    if "question_and_filler" in patch_kinds:
        sites["question_and_filler"] = _combine_patch_sites(
            sites, ["filler", "question_span"]
        )

    aligned_pairs = _aligned_source_target_pairs(
        source_prompt_ids, target_prompt_ids, response_len=response_len
    )
    excluded_control_positions: set[int] = set()
    for kind in patch_kinds:
        excluded_control_positions.update(
            int(value) for value in sites.get(kind, {}).get("source_indices", [])
        )

    patch_specs = []
    skipped = {}
    for kind in patch_kinds:
        site = sites.get(kind, {})
        source_indices = [int(value) for value in site.get("source_indices", [])]
        target_indices = [int(value) for value in site.get("target_indices", [])]
        if not source_indices or len(source_indices) != len(target_indices):
            skipped[kind] = "no aligned patch positions detected"
            continue
        patch_specs.append({
            "name": kind,
            "source_indices": source_indices,
            "target_indices": target_indices,
            "is_control": False,
            "control_for": None,
        })
        seed_material = (
            f"{int(config.get('run', {}).get('seed', 0))}|{record.get('record_id')}|{kind}|native-control"
        ).encode("utf-8")
        control_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        control_site = _random_patch_control(
            aligned_pairs=aligned_pairs,
            excluded_source_indices=excluded_control_positions,
            count=len(source_indices),
            seed=control_seed,
        )
        control_source = [int(value) for value in control_site["source_indices"]]
        control_target = [int(value) for value in control_site["target_indices"]]
        control_name = f"{kind}_random_control"
        if not control_source or len(control_source) != len(source_indices):
            skipped[control_name] = (
                "not enough non-semantic aligned positions for equal-size random control"
            )
            continue
        patch_specs.append({
            "name": control_name,
            "source_indices": control_source,
            "target_indices": control_target,
            "is_control": True,
            "control_for": kind,
        })
        sites[control_name] = {
            "source_indices": control_source,
            "target_indices": control_target,
            "random_control_for": kind,
        }

    return {
        "record": record,
        "canonical_sign": canonical_direction_sign(str(record["direction"])),
        "source_prompt_ids": source_prompt_ids,
        "target_prompt_ids": target_prompt_ids,
        "response_ids": response_ids,
        "source_sequence": source_sequence,
        "target_sequence": target_sequence,
        "support_ids": support_ids,
        "support_mask": support_mask,
        "response_len": response_len,
        "sites": sites,
        "patch_specs": patch_specs,
        "skipped": skipped,
    }


@torch.no_grad()
def _forward_source_target_record_batch(
    runtime: Runtime,
    prepared: list[dict],
) -> list[dict]:
    """Run source and target conditions for several records in one transformer pass."""
    sequences: list[torch.Tensor] = []
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []
    for item in prepared:
        sequences.extend([item["source_sequence"], item["target_sequence"]])
        prompt_lengths.extend([
            int(item["source_prompt_ids"].numel()),
            int(item["target_prompt_ids"].numel()),
        ])
        response_lengths.extend([int(item["response_len"]), int(item["response_len"])])

    input_ids, attention_mask = _right_pad_sequences(
        sequences,
        pad_token_id=_pad_token_id(runtime.tokenizer),
        device=runtime.device,
    )
    head_context = _native_response_head_batch_context(
        runtime,
        prompt_lengths=prompt_lengths,
        response_lengths=response_lengths,
    )
    with _HiddenCapture(runtime.layer_module) as capture:
        if head_context is None:
            result = runtime.model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            )
        else:
            with head_context:
                result = runtime.model(
                    input_ids=input_ids, attention_mask=attention_mask, use_cache=False
                )
    if capture.hidden is None:
        raise RuntimeError("Native analysis failed to capture the patch-layer residual")

    outputs = []
    for record_index, item in enumerate(prepared):
        source_batch_index = 2 * record_index
        target_batch_index = source_batch_index + 1
        response_len = int(item["response_len"])
        if head_context is None:
            source_logits = _response_slice(
                result.logits[source_batch_index:source_batch_index + 1],
                prompt_len=int(item["source_prompt_ids"].numel()),
                response_len=response_len,
            )
            target_logits = _response_slice(
                result.logits[target_batch_index:target_batch_index + 1],
                prompt_len=int(item["target_prompt_ids"].numel()),
                response_len=response_len,
            )
        else:
            source_logits = result.logits[source_batch_index:source_batch_index + 1, :response_len]
            target_logits = result.logits[target_batch_index:target_batch_index + 1, :response_len]
        target_valid = int(item["target_sequence"].numel())
        outputs.append({
            "target_hidden": capture.hidden[
                target_batch_index:target_batch_index + 1, :target_valid
            ].detach(),
            "source_summary": _summarize_response_logits(
                source_logits,
                support_ids=item["support_ids"],
                support_mask=item["support_mask"],
                response_ids=item["response_ids"],
            ),
            "target_summary": _summarize_response_logits(
                target_logits,
                support_ids=item["support_ids"],
                support_mask=item["support_mask"],
                response_ids=item["response_ids"],
            ),
        })
    del result, input_ids, attention_mask
    return outputs


@torch.no_grad()
def _forward_patch_record_batch(
    runtime: Runtime,
    prepared: list[dict],
    chunk: list[dict],
) -> list[dict[str, torch.Tensor]]:
    """Run heterogeneous patch conditions from several records in one padded batch."""
    if not chunk:
        return []
    sequences = [prepared[int(row["record_index"])]["source_sequence"] for row in chunk]
    input_ids, attention_mask = _right_pad_sequences(
        sequences,
        pad_token_id=_pad_token_id(runtime.tokenizer),
        device=runtime.device,
    )
    prompt_lengths = [
        int(prepared[int(row["record_index"])]["source_prompt_ids"].numel())
        for row in chunk
    ]
    response_lengths = [
        int(prepared[int(row["record_index"])]["response_len"])
        for row in chunk
    ]
    head_context = _native_response_head_batch_context(
        runtime,
        prompt_lengths=prompt_lengths,
        response_lengths=response_lengths,
    )
    with _BatchedActivationReplaceHook(
        runtime.layer_module,
        [row["source_indices"] for row in chunk],
        [row["target_values"] for row in chunk],
    ):
        if head_context is None:
            result = runtime.model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            )
        else:
            with head_context:
                result = runtime.model(
                    input_ids=input_ids, attention_mask=attention_mask, use_cache=False
                )

    summaries = []
    for batch_index, row in enumerate(chunk):
        item = prepared[int(row["record_index"])]
        response_len = int(item["response_len"])
        if head_context is None:
            response_logits = _response_slice(
                result.logits[batch_index:batch_index + 1],
                prompt_len=int(item["source_prompt_ids"].numel()),
                response_len=response_len,
            )
        else:
            response_logits = result.logits[batch_index:batch_index + 1, :response_len]
        summaries.append(_summarize_response_logits(
            response_logits,
            support_ids=item["support_ids"],
            support_mask=item["support_mask"],
            response_ids=item["response_ids"],
        ))
    del result, input_ids, attention_mask
    return summaries


def _forward_patch_record_chunk_with_oom_fallback(
    runtime: Runtime,
    prepared: list[dict],
    chunk: list[dict],
) -> list[dict[str, torch.Tensor]]:
    """Evaluate a cross-record patch chunk, recursively halving on CUDA OOM."""
    if not chunk:
        return []
    try:
        return _forward_patch_record_batch(runtime, prepared, chunk)
    except RuntimeError as exc:
        if not _cuda_oom(exc) or len(chunk) <= 1:
            raise
        if torch.cuda.is_available() and str(runtime.device).startswith("cuda"):
            torch.cuda.empty_cache()
        split = max(1, len(chunk) // 2)
        print(
            "native activation patching: CUDA OOM at condition_batch="
            f"{len(chunk)}; continuing with condition_batch<={max(split, len(chunk)-split)}",
            flush=True,
        )
        return (
            _forward_patch_record_chunk_with_oom_fallback(
                runtime, prepared, chunk[:split]
            )
            + _forward_patch_record_chunk_with_oom_fallback(
                runtime, prepared, chunk[split:]
            )
        )


def _finalize_native_record(
    config: dict,
    runtime: Runtime,
    item: dict,
    *,
    source_summary: dict[str, torch.Tensor],
    target_summary: dict[str, torch.Tensor],
    patch_summaries: dict[str, dict[str, torch.Tensor]],
    include_qualitative_probabilities: bool,
) -> dict:
    settings = config.get("native_analysis", {})
    record = item["record"]
    response_ids = item["response_ids"]
    support_ids = item["support_ids"]
    support_mask = item["support_mask"]
    response_len = int(item["response_len"])
    source_kl_metrics = _conditional_kl_metrics(record, source_summary)
    target_kl_metrics = _conditional_kl_metrics(record, target_summary)

    shifts: dict[str, dict] = {
        "intervention": _shift_details(
            record,
            support_ids=support_ids,
            support_mask=support_mask,
            before=source_summary,
            after=target_summary,
            canonical_sign=float(item["canonical_sign"]),
            tokenizer=runtime.tokenizer,
            top_tokens_per_position=int(settings.get("top_tokens_per_position", 8)),
            top_positions=int(settings.get("top_positions_per_record", 20)),
        )
    }

    patch_kl_metrics: dict[str, dict] = {}
    control_kl_metrics: dict[str, dict] = {}
    for spec in item["patch_specs"]:
        kind = str(spec["name"])
        patch_summary = patch_summaries[kind]
        metrics = _conditional_kl_metrics(record, patch_summary)
        if bool(spec.get("is_control", False)):
            control_for = str(spec.get("control_for"))
            control_kl_metrics[control_for] = {
                **metrics,
                "patch_name": kind,
                "source_indices": [int(value) for value in spec["source_indices"]],
            }
            if include_qualitative_probabilities:
                control_kl_metrics[control_for]["qualitative_probability_table"] = (
                    _native_qualitative_probability_table(
                        runtime.tokenizer,
                        support_ids,
                        support_mask,
                        source_summary=source_summary,
                        target_summary=target_summary,
                        patch_summary=patch_summary,
                        top_n=int(config.get("report", {}).get("qualitative_probability_token_limit", 20)),
                    )
                )
        else:
            patch_kl_metrics[kind] = {
                **metrics,
                "source_indices": [int(value) for value in spec["source_indices"]],
            }
            if include_qualitative_probabilities:
                patch_kl_metrics[kind]["qualitative_probability_table"] = (
                    _native_qualitative_probability_table(
                        runtime.tokenizer,
                        support_ids,
                        support_mask,
                        source_summary=source_summary,
                        target_summary=target_summary,
                        patch_summary=patch_summary,
                        top_n=int(config.get("report", {}).get("qualitative_probability_token_limit", 20)),
                    )
                )
        shifts[kind] = _shift_details(
            record,
            support_ids=support_ids,
            support_mask=support_mask,
            before=source_summary,
            after=patch_summary,
            canonical_sign=float(item["canonical_sign"]),
            tokenizer=runtime.tokenizer,
            top_tokens_per_position=int(settings.get("top_tokens_per_position", 8)),
            top_positions=int(settings.get("top_positions_per_record", 20)),
        )

    return {
        "record_id": str(record.get("record_id", "")),
        "source_id": str(record.get("source_id", "")),
        "response_id": str(record.get("response_id", "")),
        "direction": str(record.get("direction", "")),
        "response_tokens": response_len,
        "response_token_texts": [
            _display_token(runtime_decode_token(runtime.tokenizer, int(value)))
            for value in response_ids.tolist()
        ],
        "patch_sites": item["sites"],
        "skipped_patch_sites": item["skipped"],
        "native_fixed_prefix": {
            "definition": "conditional top-k KL recovery of native source/target/patch computations under the same fixed response prefix",
            "source": source_kl_metrics,
            "target": target_kl_metrics,
            "patches": patch_kl_metrics,
            "random_controls": control_kl_metrics,
        },
        "mass_shift_orientation": "toward_suggestion",
        "mass_shift": shifts,
    }


def analyze_native_records_batch(
    config: dict,
    runtime: Runtime,
    records: list[dict],
    *,
    include_qualitative_probabilities: list[bool] | None = None,
) -> list[dict]:
    """Analyze several records with true cross-record native-model batching."""
    if not records:
        return []
    flags = list(include_qualitative_probabilities or [False] * len(records))
    if len(flags) != len(records):
        raise ValueError("qualitative flag count must match native record batch size")
    prepared = [_prepare_native_record(config, runtime, record) for record in records]

    try:
        base_outputs = _forward_source_target_record_batch(runtime, prepared)
    except RuntimeError as exc:
        # For a single record, split conditions after an OOM when evaluating
        # the source/target pair. Record-batch OOM splitting is handled outside.
        if not _cuda_oom(exc) or len(prepared) != 1:
            raise
        if torch.cuda.is_available() and str(runtime.device).startswith("cuda"):
            torch.cuda.empty_cache()
        item = prepared[0]
        print(
            "native activation patching: CUDA OOM at source_target_batch=2; "
            "continuing with source/target sequentially",
            flush=True,
        )
        source_input = item["source_sequence"].to(runtime.device, dtype=torch.long).view(1, -1)
        target_input = item["target_sequence"].to(runtime.device, dtype=torch.long).view(1, -1)
        _source_hidden, source_summary = _forward_source_or_target(
            runtime,
            source_input,
            prompt_len=int(item["source_prompt_ids"].numel()),
            response_len=int(item["response_len"]),
            support_ids=item["support_ids"],
            support_mask=item["support_mask"],
            response_ids=item["response_ids"],
        )
        target_hidden, target_summary = _forward_source_or_target(
            runtime,
            target_input,
            prompt_len=int(item["target_prompt_ids"].numel()),
            response_len=int(item["response_len"]),
            support_ids=item["support_ids"],
            support_mask=item["support_mask"],
            response_ids=item["response_ids"],
        )
        del _source_hidden
        base_outputs = [{
            "target_hidden": target_hidden,
            "source_summary": source_summary,
            "target_summary": target_summary,
        }]

    patch_rows: list[dict] = []
    for record_index, (item, base) in enumerate(zip(prepared, base_outputs)):
        target_hidden = base["target_hidden"]
        for spec in item["patch_specs"]:
            target_index_tensor = torch.tensor(
                spec["target_indices"], device=target_hidden.device, dtype=torch.long
            )
            row = dict(spec)
            row["record_index"] = record_index
            row["target_values"] = target_hidden.index_select(1, target_index_tensor)
            patch_rows.append(row)

    condition_batch_size = max(
        1, int(config.get("native_analysis", {}).get("native_condition_batch_size", 16))
    )
    patch_summaries_by_record: list[dict[str, dict[str, torch.Tensor]]] = [
        {} for _ in prepared
    ]
    for start in range(0, len(patch_rows), condition_batch_size):
        chunk = patch_rows[start:start + condition_batch_size]
        summaries = _forward_patch_record_chunk_with_oom_fallback(
            runtime, prepared, chunk
        )
        for row, summary in zip(chunk, summaries):
            patch_summaries_by_record[int(row["record_index"])][str(row["name"])] = summary

    payloads = []
    for index, item in enumerate(prepared):
        payloads.append(_finalize_native_record(
            config,
            runtime,
            item,
            source_summary=base_outputs[index]["source_summary"],
            target_summary=base_outputs[index]["target_summary"],
            patch_summaries=patch_summaries_by_record[index],
            include_qualitative_probabilities=bool(flags[index]),
        ))
    return payloads


def analyze_native_record(
    config: dict,
    runtime: Runtime,
    record: dict,
    *,
    include_qualitative_probabilities: bool = False,
) -> dict:
    """Analyze one record through the batched native-patching implementation."""
    return analyze_native_records_batch(
        config,
        runtime,
        [record],
        include_qualitative_probabilities=[include_qualitative_probabilities],
    )[0]


def _native_patch_summary(config: dict, payloads: list[dict]) -> dict:
    settings = config.get("native_analysis", {})
    sites = [str(value) for value in settings.get(
        "patch_sites",
        ["filler", "question_span", "question_and_filler"],
    )]
    bootstrap_samples = int(settings.get("bootstrap_samples", 1000))
    seed = int(config.get("run", {}).get("seed", 0)) + 811
    rows = []
    for payload in payloads:
        fixed = payload.get("native_fixed_prefix", {})
        source = fixed.get("source", {})
        patches = fixed.get("patches", {})
        controls = fixed.get("random_controls", {})
        if not source:
            continue
        rows.append(
            {
                "source_id": str(payload.get("source_id", "")),
                "source_kl_mean": float(source.get("source_kl_mean", 0.0)),
                "sites": {
                    site: {
                        "condition_kl_mean": float(patches[site]["condition_kl_mean"])
                    }
                    for site in sites
                    if site in patches
                },
                "controls": {
                    site: {
                        "condition_kl_mean": float(controls[site]["condition_kl_mean"])
                    }
                    for site in sites
                    if site in controls
                },
            }
        )
    source_mean = float(np.mean([row["source_kl_mean"] for row in rows])) if rows else 0.0
    target_means = [
        float(payload.get("native_fixed_prefix", {}).get("target", {}).get("condition_kl_mean", 0.0))
        for payload in payloads
        if payload.get("native_fixed_prefix", {}).get("target")
    ]
    target_mean = float(np.mean(target_means)) if target_means else 0.0

    def aggregate(site: str, key: str, offset: int) -> dict:
        transformed = [
            {
                "source_id": row["source_id"],
                "source_kl_mean": row["source_kl_mean"],
                "sites": ({site: row[key][site]} if site in row.get(key, {}) else {}),
            }
            for row in rows
        ]
        metric = _site_metric_aggregate(
            transformed,
            site=site,
            variant_key="condition_kl_mean",
            full_key=None,
            bootstrap_samples=bootstrap_samples,
            seed=seed + offset,
        )
        return {**metric, "patch_recovery": metric.get("value")}

    return {
        "records": len(rows),
        "sources": len({str(row.get("source_id", "")) for row in rows}),
        "source_kl_mean": source_mean,
        "target_kl_mean": target_mean,
        "target_recovery": recovery_fraction(source_mean, target_mean),
        "sites": {
            site: aggregate(site, "sites", index)
            for index, site in enumerate(sites)
        },
        "random_controls": {
            site: aggregate(site, "controls", 100 + index)
            for index, site in enumerate(sites)
        },
        "control_definition": (
            "For each semantic patch, patch an equal number of deterministically random "
            "source↔target token-aligned positions sampled outside all semantic patch sites."
        ),
    }


def _compact_native_summary_payload(payload: dict) -> dict:
    """Keep only native fixed-prefix fields needed for aggregate patch statistics."""
    fixed = payload.get("native_fixed_prefix", {})
    compact_fixed = {
        "source": {
            "source_kl_mean": float(fixed.get("source", {}).get("source_kl_mean", 0.0))
        } if fixed.get("source") else {},
        "target": {
            "condition_kl_mean": float(fixed.get("target", {}).get("condition_kl_mean", 0.0))
        } if fixed.get("target") else {},
        "patches": {},
        "random_controls": {},
    }
    for key in ("patches", "random_controls"):
        for site, metric in fixed.get(key, {}).items():
            if metric.get("condition_kl_mean") is not None:
                compact_fixed[key][str(site)] = {
                    "condition_kl_mean": float(metric["condition_kl_mean"])
                }
    return {
        "source_id": str(payload.get("source_id", "")),
        "native_fixed_prefix": compact_fixed,
    }


def _iter_native_records(
    records: list[dict] | None,
    record_paths: list[Path] | None,
):
    if record_paths is not None:
        for path in record_paths:
            yield torch.load(path, map_location="cpu")
    else:
        yield from (records or [])


def _iter_native_record_batches(
    records: list[dict] | None,
    record_paths: list[Path] | None,
    *,
    batch_size: int,
):
    batch: list[dict] = []
    for record in _iter_native_records(records, record_paths):
        batch.append(record)
        if len(batch) >= int(batch_size):
            yield batch
            batch = []
    if batch:
        yield batch


def _analyze_native_record_batch_with_oom_fallback(
    config: dict,
    runtime: Runtime,
    records: list[dict],
    *,
    qualitative_record_keys: set[tuple[str, str, str]] | None,
) -> list[dict]:
    flags = [
        bool(
            qualitative_record_keys is not None
            and qualitative_record_key(record) in qualitative_record_keys
        )
        for record in records
    ]
    try:
        return analyze_native_records_batch(
            config,
            runtime,
            records,
            include_qualitative_probabilities=flags,
        )
    except RuntimeError as exc:
        if not _cuda_oom(exc) or len(records) <= 1:
            raise
        if torch.cuda.is_available() and str(runtime.device).startswith("cuda"):
            torch.cuda.empty_cache()
        split = max(1, len(records) // 2)
        print(
            "native activation patching: CUDA OOM at record_batch="
            f"{len(records)}; continuing with record_batch<={max(split, len(records)-split)}",
            flush=True,
        )
        return (
            _analyze_native_record_batch_with_oom_fallback(
                config, runtime, records[:split],
                qualitative_record_keys=qualitative_record_keys,
            )
            + _analyze_native_record_batch_with_oom_fallback(
                config, runtime, records[split:],
                qualitative_record_keys=qualitative_record_keys,
            )
        )


def _analyze_native_record_batch_isolated(
    config: dict,
    runtime: Runtime,
    records: list[dict],
    *,
    qualitative_record_keys: set[tuple[str, str, str]] | None,
) -> tuple[list[dict], list[dict]]:
    """Retry failed batches separately; only singleton failures exclude a record."""
    try:
        payloads = _analyze_native_record_batch_with_oom_fallback(
            config, runtime, records,
            qualitative_record_keys=qualitative_record_keys,
        )
        return payloads, []
    except Exception as exc:
        if len(records) <= 1:
            return [], [{
                "record_id": str(record.get("record_id", "")),
                "source_id": str(record.get("source_id", "")),
                "direction": str(record.get("direction", "")),
                "type": type(exc).__name__,
                "message": str(exc),
            } for record in records]
        print(
            f"native activation patching: {type(exc).__name__} at "
            f"record_batch={len(records)}; splitting to isolate failing records",
            flush=True,
        )
    # Leave the exception handler before retrying so failed-forward tensors
    # referenced by the traceback can be released.
    split = len(records) // 2
    left, left_failures = _analyze_native_record_batch_isolated(
        config, runtime, records[:split], qualitative_record_keys=qualitative_record_keys,
    )
    right, right_failures = _analyze_native_record_batch_isolated(
        config, runtime, records[split:], qualitative_record_keys=qualitative_record_keys,
    )
    return left + right, left_failures + right_failures


def _compute_native_hypothesis_analysis_impl(
    config: dict,
    runtime: Runtime,
    records: list[dict] | None,
    output: Path,
    *,
    qualitative_record_keys: set[tuple[str, str, str]] | None = None,
    record_paths: list[Path] | None = None,
) -> dict:
    output = Path(output)
    record_dir = output / "records"
    record_dir.mkdir(parents=True, exist_ok=True)
    settings = config.get("native_analysis", {})
    record_batch_size = max(1, int(settings.get("native_record_batch_size", 2)))
    completed = []
    failures = []
    total = len(record_paths) if record_paths is not None else len(records or [])
    progress = tqdm(total=total, desc="native activation patching")
    try:
        for batch in _iter_native_record_batches(
            records, record_paths, batch_size=record_batch_size
        ):
            try:
                payloads, batch_failures = _analyze_native_record_batch_isolated(
                    config,
                    runtime,
                    batch,
                    qualitative_record_keys=qualitative_record_keys,
                )
                failures.extend(batch_failures)
                for payload in payloads:
                    write_json(record_dir / f"{payload['record_id']}.json", payload)
                    completed.append(_compact_native_summary_payload(payload))
                del payloads
                if batch_failures:
                    break
            finally:
                progress.update(len(batch))
    finally:
        progress.close()
    summary = {
        "status": "complete" if not failures else "failed",
        "records": len(completed),
        "sources": len({str(row["source_id"]) for row in completed}),
        "failures": failures,
        "native_patching": _native_patch_summary(config, completed),
    }
    write_json(output / "summary.json", summary)
    if failures:
        raise RuntimeError(
            f"Native patching stopped after {len(failures)} isolated record failure(s); "
            f"diagnostics saved to {output / 'summary.json'}. "
            f"First failure: {failures[0]['message']}"
        )
    return summary


def compute_native_hypothesis_analysis(
    config: dict,
    runtime: Runtime,
    records: list[dict] | None,
    output: Path,
    *,
    qualitative_record_keys: set[tuple[str, str, str]] | None = None,
    record_paths: list[Path] | None = None,
) -> dict:
    return _compute_native_hypothesis_analysis_impl(
        config, runtime, records, output,
        qualitative_record_keys=qualitative_record_keys,
        record_paths=record_paths,
    )


def merge_native_hypothesis_shards(
    config: dict,
    shard_directories: list[Path],
    output: Path,
) -> dict:
    """Merge source-disjoint native-analysis shards."""
    output = Path(output)
    record_output = output / "records"
    if record_output.exists():
        shutil.rmtree(record_output)
    record_output.mkdir(parents=True, exist_ok=True)
    all_payloads = []
    failures = []
    sources = set()
    for shard in shard_directories:
        for path in sorted((Path(shard) / "records").glob("*.json")):
            target = record_output / path.name
            shutil.copy2(path, target)
            payload = json.loads(path.read_text(encoding="utf-8"))
            all_payloads.append(_compact_native_summary_payload(payload))
            sources.add(str(payload.get("source_id", "")))
        shard_summary_path = Path(shard) / "summary.json"
        if shard_summary_path.is_file():
            shard_summary = json.loads(shard_summary_path.read_text(encoding="utf-8"))
            failures.extend(shard_summary.get("failures", []))
    summary = {
        "status": "complete" if not failures else "failed",
        "records": len(all_payloads),
        "sources": len(sources),
        "failures": failures,
        "native_patching": _native_patch_summary(config, all_payloads),
    }
    write_json(output / "summary.json", summary)
    if failures:
        raise RuntimeError(
            f"Native patching has {len(failures)} isolated record failure(s); "
            f"diagnostics saved to {output / 'summary.json'}. "
            f"First failure: {failures[0]['message']}"
        )
    return summary


def compute_native_hypothesis_analysis_shard(
    config: dict,
    runtime: Runtime,
    records: list[dict] | None,
    output: Path,
    *,
    qualitative_record_keys: set[tuple[str, str, str]] | None = None,
    record_paths: list[Path] | None = None,
) -> dict:
    """Compute a source-disjoint shard for later deterministic merging."""
    return _compute_native_hypothesis_analysis_impl(
        config, runtime, records, output,
        qualitative_record_keys=qualitative_record_keys,
        record_paths=record_paths,
    )
