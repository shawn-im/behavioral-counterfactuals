from __future__ import annotations

"""Internal ablation implementation for :mod:`behavior_pipeline.counterfactual_runtime`."""

import hashlib
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm.auto import tqdm

from .topk_kl import (
    conditional_topk_kl,
    conditional_logp_from_support_logits,
    logp_on_support_from_logits,
    mean_kl_increase,
)
from .statistics import pearson_correlation, spearman_correlation
from .token_text import token_id_texts
from .counterfactual_runtime_core import (
    Runtime,
    _signed_phrase_spans,
    read_json,
    write_json,
)
from .counterfactual_runtime_attribution import (
    _attribution_filename,
    qualitative_record_key,
    select_qualitative_record_keys,
)
from .counterfactual_runtime_tracing import _record_paths, _response_logits, _response_support_logits


@torch.no_grad()
def score_edits(
    runtime: Runtime,
    record: dict,
    edit: torch.Tensor,
    *,
    support_only_projection: bool = False,
    support_projection_chunk_size: int = 64,
) -> dict:
    """Score explicit applied-edit trajectories without reconstructing editor features."""
    if edit.ndim != 3:
        raise ValueError(f"edit must have shape [batch, sequence, d_model], got {tuple(edit.shape)}")
    batch_size = int(edit.shape[0])
    prompt_ids = record["prompt_ids"].to(runtime.device, dtype=torch.long)
    response_ids = record["response_ids"].to(runtime.device, dtype=torch.long)
    input_ids = torch.cat([prompt_ids, response_ids]).view(1, -1).expand(batch_size, -1)
    edit = edit.to(runtime.device, dtype=torch.float32)
    support_ids_base = record["support_ids"].to(runtime.device, dtype=torch.long)
    support_mask = record["support_mask"].to(runtime.device, dtype=torch.bool).unsqueeze(0).expand(batch_size, -1, -1)
    target_logp = record["target_logp"].to(runtime.device).float().unsqueeze(0).expand(batch_size, -1, -1)
    if support_only_projection:
        support_logits = _response_support_logits(
            runtime, input_ids, prompt_len=int(prompt_ids.numel()),
            response_len=int(response_ids.numel()), edit=edit,
            support_ids=support_ids_base, chunk_size=int(support_projection_chunk_size),
            requires_grad=False,
        )
        logp = conditional_logp_from_support_logits(support_logits, support_mask)
        losses = conditional_topk_kl(target_logp, logp, support_mask)
        support_mass = torch.full_like(losses, float("nan"))
    else:
        logits = _response_logits(
            runtime, input_ids, prompt_len=int(prompt_ids.numel()),
            response_len=int(response_ids.numel()), edit=edit,
        )
        support_ids = support_ids_base.unsqueeze(0).expand(batch_size, -1, -1)
        logp, support_mass = logp_on_support_from_logits(logits, support_ids, support_mask)
        losses = conditional_topk_kl(target_logp, logp, support_mask)
    return {
        "logp": logp.detach().cpu(), "loss": losses.detach().cpu(),
        "support_mass": support_mass.detach().cpu(), "edit": edit.detach().cpu(),
    }


def _qualitative_support_probability_table(
    tokenizer,
    support_ids: torch.Tensor,
    support_mask: torch.Tensor,
    *,
    source_logp: torch.Tensor,
    target_logp: torch.Tensor,
    edited_logp: torch.Tensor,
    ablated_logp: torch.Tensor,
    top_n: int = 20,
) -> list[dict]:
    """Compact full-softmax support probabilities for HTML example callouts.

    We keep this only for the deterministic qualitative subset.  At each
    response position alternatives are ranked by absolute edited-minus-ablated
    probability change, so the saved shortlist contains the largest changes.
    """
    support_ids = support_ids.detach().long().cpu()
    support_mask = support_mask.detach().bool().cpu()
    conditions = {
        "source": source_logp.detach().float().cpu().exp(),
        "target": target_logp.detach().float().cpu().exp(),
        "edited": edited_logp.detach().float().cpu().exp(),
        "edit_ablated": ablated_logp.detach().float().cpu().exp(),
    }
    rows: list[dict] = []
    response_len = int(support_ids.shape[0])
    for response_index in range(response_len):
        valid = torch.nonzero(support_mask[response_index], as_tuple=False).flatten()
        if valid.numel() == 0:
            rows.append({"response_index": response_index, "alternatives": []})
            continue
        matrix = torch.stack(
            [conditions[name][response_index, valid] for name in conditions], dim=0
        )
        separation = (conditions["edited"][response_index, valid]
                      - conditions["edit_ablated"][response_index, valid]).abs()
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


def _record_with_matched_baseline(record: dict, scored: dict) -> dict:
    """Use a full-edit rerun from the same batch as the intervention.

    Batched bf16 Transformer execution is not bitwise invariant to padding and
    batch shape. Causal effects therefore need a matched full-edit rerun rather
    than the logits recorded by an earlier, differently shaped trace batch.
    """
    matched = dict(record)
    matched["edited_logp"] = scored["logp"].float()
    matched["edited_kl"] = scored["loss"].float()
    matched["support_mass"] = scored["support_mass"].float()
    matched["edited_kl_mean"] = float(scored["loss"].float().mean())
    return matched


def _baseline_parity(record: dict, matched: dict) -> dict:
    recorded = record["edited_kl"].float()
    rerun = matched["edited_kl"].float()
    return {
        "recorded_full_kl_mean": float(recorded.mean()),
        "matched_full_kl_mean": float(rerun.mean()),
        "baseline_max_token_kl_error": float((rerun - recorded).abs().max()),
    }


def _load_total_attribution(path: Path) -> dict[str, dict]:
    payload = read_json(path, {})
    return {
        str(row["record_id"]): row for row in payload.get("records", [])
    }


def _correlation_summary(
    rows: list[dict],
    *,
    predicted_key: str,
    observed_key: str,
) -> dict:
    pairs = [
        (float(row[predicted_key]), float(row[observed_key]))
        for row in rows
        if row.get(predicted_key) is not None
        and row.get(observed_key) is not None
        and math.isfinite(float(row[predicted_key]))
        and math.isfinite(float(row[observed_key]))
    ]
    if not pairs:
        return {
            "count": 0,
            "pearson": None,
            "spearman": None,
            "sign_accuracy": None,
        }
    predicted = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    observed = np.asarray([pair[1] for pair in pairs], dtype=np.float64)

    nonzero = (predicted != 0.0) | (observed != 0.0)
    return {
        "count": int(len(pairs)),
        "pearson": pearson_correlation(predicted, observed),
        "spearman": spearman_correlation(predicted, observed),
        "sign_accuracy": (
            float(
                np.mean(
                    np.sign(predicted[nonzero])
                    == np.sign(observed[nonzero])
                )
            )
            if bool(nonzero.any())
            else None
        ),
    }


def _source_clustered_causal_summary(
    rows: list[dict],
    *,
    is_primary,
    seed: int,
    predicted_key: str = "attribution_credit",
    observed_key: str = "mean_kl_increase",
    is_control=None,
    bootstrap_samples: int = 2000,
) -> dict:
    """Summarize rank agreement and matched effects with source clustering."""
    by_record: dict[str, list[dict]] = defaultdict(list)
    source_by_record: dict[str, str] = {}
    for row in rows:
        if row.get(predicted_key) is None or row.get(observed_key) is None:
            continue
        record_id = str(row["record_id"])
        by_record[record_id].append(row)
        source_by_record[record_id] = str(row.get("source_id", record_id))

    source_ranks: dict[str, list[float]] = defaultdict(list)
    source_differences: dict[str, list[float]] = defaultdict(list)
    source_primary_abs: dict[str, list[float]] = defaultdict(list)
    source_control_abs: dict[str, list[float]] = defaultdict(list)
    for record_id, record_rows in by_record.items():
        summary = _correlation_summary(
            record_rows, predicted_key=predicted_key, observed_key=observed_key
        )
        source_id = source_by_record[record_id]
        if summary.get("spearman") is not None:
            source_ranks[source_id].append(float(summary["spearman"]))
        primary = [abs(float(row[observed_key])) for row in record_rows if is_primary(row)]
        controls = [
            abs(float(row[observed_key]))
            for row in record_rows
            if (is_control(row) if is_control is not None else not is_primary(row))
        ]
        if primary:
            source_primary_abs[source_id].append(float(np.mean(primary)))
        if controls:
            source_control_abs[source_id].append(float(np.mean(controls)))
        if primary and controls:
            source_differences[source_id].append(float(np.mean(primary) - np.mean(controls)))

    rank_values = np.asarray([float(np.mean(values)) for values in source_ranks.values()], dtype=np.float64)
    difference_values = np.asarray([float(np.mean(values)) for values in source_differences.values()], dtype=np.float64)
    primary_values = np.asarray([float(np.mean(values)) for values in source_primary_abs.values()], dtype=np.float64)
    control_values = np.asarray([float(np.mean(values)) for values in source_control_abs.values()], dtype=np.float64)
    rng = random.Random(int(seed))
    paired_bootstrap, rank_bootstrap = [], []
    if difference_values.size:
        for _ in range(int(bootstrap_samples)):
            sample = [float(difference_values[rng.randrange(len(difference_values))]) for _ in range(len(difference_values))]
            paired_bootstrap.append(float(np.mean(sample)))
    if rank_values.size:
        for _ in range(int(bootstrap_samples)):
            sample = [float(rank_values[rng.randrange(len(rank_values))]) for _ in range(len(rank_values))]
            rank_bootstrap.append(float(np.median(sample)))

    def q(values, p):
        return float(np.quantile(values, p)) if values.size else None

    return {
        "rank_source_ids": int(rank_values.size),
        "paired_control_source_ids": int(difference_values.size),
        "median_source_level_within_record_spearman": q(rank_values, 0.5),
        "mean_source_level_within_record_spearman": float(np.mean(rank_values)) if rank_values.size else None,
        "source_level_spearman_p25": q(rank_values, 0.25),
        "source_level_spearman_p75": q(rank_values, 0.75),
        "fraction_sources_positive_spearman": float(np.mean(rank_values > 0.0)) if rank_values.size else None,
        "median_spearman_bootstrap_95_ci": ([float(np.quantile(rank_bootstrap, 0.025)), float(np.quantile(rank_bootstrap, 0.975))] if rank_bootstrap else None),
        "paired_top_minus_control_abs_mean_kl_increase": float(np.mean(difference_values)) if difference_values.size else None,
        "median_top_minus_control_abs_mean_kl_increase": q(difference_values, 0.5),
        "fraction_sources_top_abs_effect_gt_control": float(np.mean(difference_values > 0.0)) if difference_values.size else None,
        "mean_source_primary_abs_effect": float(np.mean(primary_values)) if primary_values.size else None,
        "mean_source_control_abs_effect": float(np.mean(control_values)) if control_values.size else None,
        "paired_bootstrap_95_ci": ([float(np.quantile(paired_bootstrap, 0.025)), float(np.quantile(paired_bootstrap, 0.975))] if paired_bootstrap else None),
        "source_level_spearman_values": rank_values.tolist(),
        "source_level_top_minus_control_values": difference_values.tolist(),
    }


def _is_method_evaluation(row: dict, method: str) -> bool:
    return _method_selection_group(row, method) in {"top", "random"}


def _is_method_signed_top(row: dict, method: str) -> bool:
    metadata = row.get("selection_groups", {}).get(str(method), {})
    if bool(metadata.get("signed_top", False)):
        return True
    return f"{method}_signed_top" in {
        str(value) for value in row.get("selection_sources", [])
    }


def _method_selection_group(row: dict, method: str) -> str | None:
    metadata = row.get("selection_groups", {}).get(str(method), {})
    group = metadata.get("evaluation_group", metadata.get("group"))
    if group in {"top", "random"}:
        return str(group)
    sources = {str(value) for value in row.get("selection_sources", [])}
    for candidate in ("top", "random"):
        if f"{method}_{candidate}" in sources:
            return candidate
    return None


def _causal_effect_size_distribution(rows: list[dict], method: str) -> dict:
    """Aggregate top/random causal effect magnitude once per response."""
    by_record: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"top": [], "random": []}
    )
    for row in rows:
        group = _method_selection_group(row, method)
        if group is None:
            continue
        effect_size = row.get("causal_effect_size")
        if effect_size is None:
            effect_size = abs(float(row.get("mean_kl_increase", 0.0)))
        if math.isfinite(float(effect_size)):
            by_record[str(row["record_id"])][group].append(float(effect_size))

    values: dict[str, list[float]] = {"top": [], "random": []}
    record_rows = []
    for record_id in sorted(by_record):
        record_summary = {"record_id": record_id}
        for group in ("top", "random"):
            group_values = by_record[record_id][group]
            mean_value = float(np.mean(group_values)) if group_values else None
            record_summary[f"{group}_mean_causal_effect_size"] = mean_value
            record_summary[f"{group}_events"] = len(group_values)
            if mean_value is not None:
                values[group].append(mean_value)
        record_rows.append(record_summary)

    def summarize(group: str) -> dict:
        array = np.asarray(values[group], dtype=np.float64)
        return {
            "responses": int(array.size),
            "mean": float(np.mean(array)) if array.size else None,
            "median": float(np.median(array)) if array.size else None,
            "values": array.tolist(),
        }

    return {
        "unit": "absolute mean conditional KL increase per response token",
        "top": summarize("top"),
        "random": summarize("random"),
        "by_record": record_rows,
    }


def _per_response_correlation_summary(
    rows: list[dict],
    *,
    predicted_key: str,
    observed_key: str,
) -> dict:
    spearman_values: list[float] = []
    pearson_values: list[float] = []
    for record_id in sorted({str(row["record_id"]) for row in rows}):
        record_rows = [row for row in rows if str(row["record_id"]) == record_id]
        summary = _correlation_summary(
            record_rows,
            predicted_key=predicted_key,
            observed_key=observed_key,
        )
        if summary.get("spearman") is not None and math.isfinite(float(summary["spearman"])):
            spearman_values.append(float(summary["spearman"]))
        if summary.get("pearson") is not None and math.isfinite(float(summary["pearson"])):
            pearson_values.append(float(summary["pearson"]))
    return {
        "responses_with_spearman": len(spearman_values),
        "median_spearman": float(np.median(spearman_values)) if spearman_values else None,
        "mean_spearman": float(np.mean(spearman_values)) if spearman_values else None,
        "fraction_positive_spearman": (
            float(np.mean(np.asarray(spearman_values) > 0.0))
            if spearman_values else None
        ),
        "spearman_values": spearman_values,
        "responses_with_pearson": len(pearson_values),
        "median_pearson": float(np.median(pearson_values)) if pearson_values else None,
        "pearson_values": pearson_values,
    }


def _build_edit_event_ablation_artifact(
    rows: list[dict],
    config: dict,
) -> dict:
    for row in rows:
        if row.get("causal_effect_size") is None and row.get("mean_kl_increase") is not None:
            row["causal_effect_size"] = abs(float(row["mean_kl_increase"]))
    methods = [str(value) for value in config.get("attribution", {}).get("methods", ["ig"])]
    primary_method = str(config.get("attribution", {}).get("primary_method", methods[0]))
    top_events = int(config.get("ablations", {}).get("top_events_per_method", 5))
    random_events = int(config.get("ablations", {}).get("random_events_per_method", 5))
    by_record = []
    validation_by_method: dict[str, dict] = {}
    for method in methods:
        total_key = f"{method}_total_attribution_credit"
        method_primary = [row for row in rows if _is_method_evaluation(row, method)]
        attribution_correlation = _correlation_summary(
            method_primary,
            predicted_key=total_key,
            observed_key="mean_kl_increase",
        )
        edit_norm_correlation = _correlation_summary(
            method_primary,
            predicted_key="direct_edit_norm",
            observed_key="causal_effect_size",
        )
        aggregate = {
            "attribution_vs_ablation": attribution_correlation,
            "edit_norm_vs_ablation": edit_norm_correlation,
        }
        source_validation = _source_clustered_causal_summary(
            method_primary,
            is_primary=lambda _row: True,
            is_control=lambda _row: False,
            seed=int(config["run"]["seed"]) + 17,
            predicted_key=total_key,
            observed_key="mean_kl_increase",
        )
        edit_norm_source_validation = _source_clustered_causal_summary(
            method_primary,
            is_primary=lambda _row: True,
            is_control=lambda _row: False,
            seed=int(config["run"]["seed"]) + 18,
            predicted_key="direct_edit_norm",
            observed_key="causal_effect_size",
        )
        per_response_validation = _per_response_correlation_summary(
            method_primary,
            predicted_key=total_key,
            observed_key="mean_kl_increase",
        )
        edit_norm_per_response_validation = _per_response_correlation_summary(
            method_primary,
            predicted_key="direct_edit_norm",
            observed_key="causal_effect_size",
        )
        validation_by_method[method] = {
            "aggregate": aggregate,
            "per_response_validation": per_response_validation,
            "edit_norm_per_response_validation": edit_norm_per_response_validation,
            "source_clustered_validation": source_validation,
            "edit_norm_source_clustered_validation": edit_norm_source_validation,
            "causal_effect_size_distribution": _causal_effect_size_distribution(
                method_primary, method
            ),
        }

    for record_id in sorted({str(row["record_id"]) for row in rows}):
        record_rows = [row for row in rows if str(row["record_id"]) == record_id]
        by_record.append(
            {
                "record_id": record_id,
                "methods": {
                    method: {
                        "total_response": _correlation_summary(
                            [row for row in record_rows if _is_method_evaluation(row, method)],
                            predicted_key=f"{method}_total_attribution_credit",
                            observed_key="mean_kl_increase",
                        ),
                        "edit_norm": _correlation_summary(
                            [row for row in record_rows if _is_method_evaluation(row, method)],
                            predicted_key="direct_edit_norm",
                            observed_key="causal_effect_size",
                        ),
                    }
                    for method in methods
                },
            }
        )

    return {
        "interpretation_object": "applied_edit_event",
        "attribution_methods": methods,
        "primary_attribution_method": primary_method,
        "top_events_per_method": top_events,
        "random_events_per_method": random_events,
        "evaluated_events_by_method": {
            method: sum(_is_method_evaluation(row, method) for row in rows)
            for method in methods
        },
        "signed_top_events_by_method": {
            method: sum(_is_method_signed_top(row, method) for row in rows)
            for method in methods
        },
        "effect_unit": "mean conditional KL increase per response token",
        "causal_effect_size_unit": "absolute mean conditional KL increase per response token",
        "intervention": "Set one applied edit event z_t to zero while leaving all other edit events fixed.",
        "selection_contract": "Each attribution method independently selects the five largest-magnitude total-response attribution events and samples five additional events uniformly without replacement from the remaining positions for quantitative validation. A separate five-event set is selected by largest signed attribution for HTML report examples. Overlapping events are ablated once as a compute optimization. Only the magnitude-top and random events enter validation: attribution is compared with signed mean KL increase, while edit norm is compared with absolute mean KL increase.",
        "records": rows,
        "by_record": by_record,
        "validation_by_method": validation_by_method,
    }


def _multi_method_ablation_candidates(
    record: dict,
    total_rows: dict[str, dict],
    *,
    methods: list[str],
    primary_method: str,
    top_events: int,
    random_events: int,
    seed: int,
) -> list[dict]:
    """Select top-magnitude and random non-top events for each method.

    The union is returned only to avoid rerunning an overlapping exact
    ablation. ``selection_sources`` and ``selection_groups`` retain the
    method-specific populations used by downstream statistics.
    """
    edit_tensor = record.get("edit")
    if edit_tensor is None:
        raise RuntimeError("Edit-event ablation requires an applied edit trajectory")
    sequence = int(edit_tensor.shape[1])
    edit_norm = edit_tensor[0].detach().float().norm(dim=-1).cpu()
    total_credit: dict[str, torch.Tensor] = {}
    selection_sources: dict[int, list[str]] = defaultdict(list)
    selection_groups: dict[int, dict[str, dict]] = defaultdict(dict)
    primary_indices: list[int] = []

    for method in methods:
        row = total_rows.get(method)
        if row is None:
            continue
        total = torch.tensor(
            row.get("edit_event_credit", []),
            dtype=torch.float32,
        )
        if total.ndim != 1 or int(total.numel()) != sequence:
            raise RuntimeError(
                f"{method} total-response attribution is not aligned to the {sequence}-token trace"
            )
        total_credit[method] = total
        ranked = sorted(
            range(sequence),
            key=lambda index: (-abs(float(total[index])), int(index)),
        )
        top_indices = ranked[: min(max(0, int(top_events)), sequence)]
        signed_top_indices = sorted(
            range(sequence),
            key=lambda index: (-float(total[index]), int(index)),
        )[: min(5, sequence)]
        remaining = ranked[len(top_indices):]
        method_seed = int.from_bytes(
            hashlib.sha256(f"{int(seed)}|{method}".encode("utf-8")).digest()[:8],
            "big",
        )
        method_rng = random.Random(method_seed)
        random_count = min(max(0, int(random_events)), len(remaining))
        random_indices = method_rng.sample(remaining, random_count)
        for rank, index in enumerate(top_indices, 1):
            label = f"{method}_top"
            selection_sources[index].append(label)
            selection_groups[index][method] = {
                "evaluation_group": "top",
                "rank_by_absolute_attribution": int(rank),
            }
            if index not in primary_indices:
                primary_indices.append(index)
        for index in random_indices:
            selection_sources[index].append(f"{method}_random")
            selection_groups[index][method] = {
                "evaluation_group": "random",
                "rank_by_absolute_attribution": int(ranked.index(index) + 1),
            }
            if index not in primary_indices:
                primary_indices.append(index)
        for signed_rank, index in enumerate(signed_top_indices, 1):
            selection_sources[index].append(f"{method}_signed_top")
            metadata = selection_groups[index].setdefault(method, {})
            metadata["signed_top"] = True
            metadata["signed_rank"] = int(signed_rank)
            if index not in primary_indices:
                primary_indices.append(index)

    if not primary_indices:
        return []
    def candidate(index: int) -> dict:
        row = {
            "edit_index": int(index),
            "role": "sampled_event",
            "selection_sources": list(selection_sources.get(index, [])),
            "selection_groups": dict(selection_groups.get(index, {})),
            "direct_edit_norm": float(edit_norm[index]),
        }
        for method in methods:
            total = total_credit.get(method)
            row[f"{method}_total_attribution_credit"] = (
                float(total[index]) if total is not None else None
            )
        primary_total = row.get(f"{primary_method}_total_attribution_credit")
        row["attribution_credit"] = float(primary_total or 0.0)
        row["total_attribution_credit"] = float(primary_total or 0.0)
        return row

    return [candidate(index) for index in primary_indices]


def prepare_ablation_record_shards(
    config: dict,
    record_directory: Path,
    *,
    shard_count: int,
) -> list[list[Path]]:
    """Plan stage-3 record shards using lightweight JSON sidecars.

    The parent process balances sequence lengths and returns paths for workers
    to load one at a time during causal evaluation.
    """
    maximum_records = int(config.get("ablations", {}).get("max_records", 0))
    count = max(1, int(shard_count))
    paths = _record_paths(
        record_directory,
        maximum=maximum_records,
        shard_index=0,
        shard_count=1,
    )
    weighted: list[tuple[int, Path]] = []
    for path in tqdm(
        paths,
        desc="index ablation records",
        unit="record",
        leave=False,
    ):
        metadata = read_json(path.with_suffix(".json"), {})
        cost = max(
            1,
            int(metadata.get("prompt_tokens", 0))
            + int(metadata.get("response_tokens", 0)),
        )
        weighted.append((cost, path))

    if count == 1:
        return [[
            path for _, path in sorted(weighted, key=lambda item: (item[0], item[1].name))
        ]]

    shard_weighted: list[list[tuple[int, Path]]] = [[] for _ in range(count)]
    shard_loads = [0 for _ in range(count)]
    for cost, path in sorted(weighted, key=lambda item: (-item[0], item[1].name)):
        destination = min(
            range(count),
            key=lambda index: (shard_loads[index], index),
        )
        shard_weighted[destination].append((int(cost), path))
        shard_loads[destination] += int(cost)

    return [
        [
            path
            for _, path in sorted(
                shard_weighted[index], key=lambda item: (item[0], item[1].name)
            )
        ]
        for index in range(count)
    ]


def compute_edit_event_ablations(
    runtime: Runtime,
    records: Iterable[dict] | None,
    attribution_dir: Path,
    output_dir: Path,
    config: dict,
    *,
    record_paths: list[Path] | None = None,
    candidate_indices_by_record: dict[str, list[int]] | None = None,
    qualitative_record_keys: set[tuple[str, str, str]] | None = None,
) -> dict:
    """Evaluate method-specific top-magnitude and random edit-event samples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = [str(value) for value in config.get("attribution", {}).get("methods", ["ig"])]
    primary_method = str(config.get("attribution", {}).get("primary_method", methods[0]))
    total_maps = {
        method: _load_total_attribution(
            attribution_dir / _attribution_filename(method, "total_response_credit.json")
        )
        for method in methods
    }
    top_events = int(config["ablations"].get("top_events_per_method", 5))
    random_events = int(config["ablations"].get("random_events_per_method", 5))
    batch_size = int(config["ablations"].get("batch_size", 1))
    support_only_projection = bool(config["ablations"].get("support_only_projection", False))
    support_projection_chunk_size = int(config["ablations"].get("support_projection_chunk_size", 64))
    max_span = int(config["phrases"]["max_span_tokens"])
    top_phrase_count = max(int(config["phrases"]["top_positive"]), int(config["phrases"]["top_negative"]))
    rows: list[dict] = []
    if record_paths is not None and records is not None:
        raise ValueError("provide either records or record_paths, not both")
    if record_paths is None and records is None:
        raise ValueError("records or record_paths is required")

    if qualitative_record_keys is not None:
        qualitative_keys = set(qualitative_record_keys)
    else:
        if record_paths is not None:
            # Avoid deserializing the tensor traces merely to choose the small
            # qualitative subset.  Harness callers pass the globally selected
            # keys, so this is mainly a safe fallback for programmatic use.
            summaries = [read_json(path.with_suffix(".json"), {}) for path in record_paths]
            qualitative_keys = select_qualitative_record_keys(
                summaries,
                fraction=float(config.get("attribution", {}).get("qualitative_fraction", 0.01)),
                seed=int(config.get("run", {}).get("seed", 0)),
            )
        else:
            record_list = list(records or [])
            records = record_list
            qualitative_keys = select_qualitative_record_keys(
                record_list,
                fraction=float(config.get("attribution", {}).get("qualitative_fraction", 0.01)),
                seed=int(config.get("run", {}).get("seed", 0)),
            )

    if record_paths is not None:
        record_iterator = (torch.load(path, map_location="cpu") for path in record_paths)
        record_total = len(record_paths)
    else:
        record_iterator = iter(records or [])
        record_total = len(records) if hasattr(records, "__len__") else None

    for record in tqdm(
        record_iterator,
        total=record_total,
        desc="edit-event ablations",
        unit="record",
    ):
        record_id = str(record["record_id"])
        totals_for_record = {method: total_maps[method].get(record_id) for method in methods}
        if all(row is None for row in totals_for_record.values()):
            continue
        explicit_indices = (
            candidate_indices_by_record.get(record_id)
            if candidate_indices_by_record is not None else None
        )
        if explicit_indices is None:
            seed_material = f"{int(config['run']['seed'])}|{record_id}".encode("utf-8")
            record_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
            candidates = _multi_method_ablation_candidates(
                record,
                totals_for_record,
                methods=methods,
                primary_method=primary_method,
                top_events=top_events,
                random_events=random_events,
                seed=record_seed,
            )
        else:
            # Explicit candidate mode is retained for programmatic callers.  No
            # automatic controls are added because the caller owns the set.
            edit_norm = record["edit"][0].float().norm(dim=-1)
            candidates = []
            for index in explicit_indices:
                if not 0 <= int(index) < int(edit_norm.numel()):
                    continue
                candidate = {
                    "edit_index": int(index),
                    "role": "explicit",
                    "selection_sources": [f"{method}_explicit" for method in methods],
                    "selection_groups": {
                        method: {"group": "explicit"} for method in methods
                    },
                    "direct_edit_norm": float(edit_norm[index]),
                }
                for method in methods:
                    total_row = totals_for_record.get(method)
                    total = torch.tensor(
                        (total_row or {}).get("edit_event_credit", []),
                        dtype=torch.float32,
                    )
                    candidate[f"{method}_total_attribution_credit"] = float(total[index]) if int(total.numel()) > int(index) else None
                candidate["attribution_credit"] = float(candidate.get(f"{primary_method}_total_attribution_credit") or 0.0)
                candidate["total_attribution_credit"] = candidate["attribution_credit"]
                candidates.append(candidate)

        baseline_edit = record["edit"].float()
        qualitative_ablated_logp: dict[int, torch.Tensor] = {}
        if qualitative_record_key(record) in qualitative_keys:
            qualitative_candidates = [
                candidate for candidate in candidates
                if _is_method_signed_top(candidate, "ig")
            ]
            if qualitative_candidates:
                # Conditional-KL ablations may use the support-only projection
                # fast path.  Report likelihood tables require true full-softmax
                # probabilities, so rescore only the tiny qualitative subset.
                # Use batch=1 here because full-vocabulary response logits are
                # exactly the memory-heavy object this fast path normally avoids.
                for candidate in qualitative_candidates:
                    qualitative_edit = baseline_edit.clone()
                    qualitative_edit[0, int(candidate["edit_index"]), :] = 0.0
                    qualitative_scores = score_edits(
                        runtime,
                        record,
                        qualitative_edit,
                        support_only_projection=False,
                        support_projection_chunk_size=support_projection_chunk_size,
                    )
                    qualitative_ablated_logp[int(candidate["edit_index"])] = (
                        qualitative_scores["logp"][0]
                    )
        for start in range(0, len(candidates), batch_size):
            candidate_batch = candidates[start : start + batch_size]
            edits = baseline_edit.expand(len(candidate_batch) + 1, -1, -1).clone()
            for local_index, candidate in enumerate(candidate_batch):
                edits[local_index + 1, int(candidate["edit_index"]), :] = 0.0
            scored_batch = score_edits(
                runtime,
                record,
                edits,
                support_only_projection=support_only_projection,
                support_projection_chunk_size=support_projection_chunk_size,
            )
            baseline = {key: value[0] for key, value in scored_batch.items()}
            matched_record = _record_with_matched_baseline(record, baseline)
            parity = _baseline_parity(record, matched_record)
            for local_index, candidate in enumerate(candidate_batch):
                loss = scored_batch["loss"][local_index + 1].float()
                effect = loss - matched_record["edited_kl"].float()
                edit_index = int(candidate["edit_index"])
                token = record["tokens"][edit_index]
                prompt_len = int(record["prompt_tokens"])
                full_effect = torch.cat([torch.zeros(prompt_len, dtype=torch.float32), effect.cpu()])
                effect_spans = _signed_phrase_spans(
                    record, full_effect, max_span_tokens=max_span, top_n=top_phrase_count
                )
                row = {
                    "record_id": record_id,
                    "source_id": str(record["source_id"]),
                    "direction": str(record["direction"]),
                    "mode": str(record.get("mode", "teacher_forced")),
                    **parity,
                    "edit_index": edit_index,
                    "region": "prompt" if edit_index < prompt_len else "response",
                    "region_index": edit_index if edit_index < prompt_len else edit_index - prompt_len,
                    "token_id": int(token["token_id"]),
                    "token_text": str(token["token_text"]),
                    "role": str(candidate["role"]),
                    "selection_sources": list(candidate.get("selection_sources", [])),
                    "selection_groups": dict(candidate.get("selection_groups", {})),
                    "direct_edit_norm": float(candidate.get("direct_edit_norm", 0.0)),
                    "removed_edit_rms": float(record["edit"][0, edit_index].float().square().mean().sqrt()),
                    "removed_edit_max_abs": float(record["edit"][0, edit_index].float().abs().max()),
                    "mean_kl_increase": float(mean_kl_increase(matched_record["edited_kl"], loss)),
                }
                row["causal_effect_size"] = abs(float(row["mean_kl_increase"]))
                for method in methods:
                    row[f"{method}_total_attribution_credit"] = candidate.get(f"{method}_total_attribution_credit")
                row["attribution_credit"] = float(candidate.get("attribution_credit", 0.0))
                row["total_attribution_credit"] = float(candidate.get("total_attribution_credit", 0.0))
                row["response_token_kl_increase"] = effect.tolist()
                row["downstream_effect_spans"] = effect_spans
                if (
                    qualitative_record_key(record) in qualitative_keys
                    and _is_method_signed_top(candidate, "ig")
                ):
                    row["qualitative_probability_table"] = _qualitative_support_probability_table(
                        runtime.tokenizer,
                        record["support_ids"],
                        record["support_mask"],
                        source_logp=record["source_logp"],
                        target_logp=record["target_logp"],
                        edited_logp=record["edited_logp"],
                        ablated_logp=qualitative_ablated_logp.get(
                            int(candidate["edit_index"]),
                            scored_batch["logp"][local_index + 1],
                        ),
                        top_n=int(config.get("report", {}).get("qualitative_probability_token_limit", 20)),
                    )
                rows.append(row)

    artifact = _build_edit_event_ablation_artifact(rows, config)
    write_json(output_dir / "edit_event_ablation.json", artifact)
    return artifact


def merge_edit_event_ablation_shards(
    shard_directories: list[Path],
    output_dir: Path,
    config: dict,
) -> dict:
    """Merge disjoint causal-ablation record shards and recompute summaries."""
    rows: list[dict] = []
    for directory in shard_directories:
        artifact = read_json(directory / "edit_event_ablation.json", {})
        rows.extend(artifact.get("records", []))
    artifact = _build_edit_event_ablation_artifact(rows, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "edit_event_ablation.json", artifact)
    return artifact
