from __future__ import annotations

"""Internal attribution implementation for :mod:`behavior_pipeline.counterfactual_runtime`."""

import hashlib
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm.auto import tqdm

from .topk_kl import (
    conditional_topk_kl_raw,
    conditional_logp_from_support_logits,
    logp_on_support_from_logits,
    mean_recovered_utility,
)
from .counterfactual_runtime_core import (
    Runtime,
    read_json,
    write_json,
)
from .relp import multi_scalar_relevance_patching
from .counterfactual_runtime_tracing import (
    _record_paths,
    _response_logits,
    _response_support_logits,
)


def _cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _adaptive_cuda_call(function, *, batch_size: int, device: str, label: str):
    """Run ``function(batch_size)`` and halve the batch on CUDA OOM."""
    active = max(1, int(batch_size))
    while True:
        try:
            return function(active), active
        except RuntimeError as exc:
            if not _cuda_oom(exc) or active <= 1:
                raise
            previous = active
            active = max(1, active // 2)
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.empty_cache()
            tqdm.write(
                f"{label}: CUDA OOM at batch={previous}; continuing with batch={active}"
            )


def _gauss_legendre_nodes(steps: int, *, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    nodes, weights = np.polynomial.legendre.leggauss(int(steps))
    alpha = torch.tensor((nodes + 1.0) / 2.0, dtype=torch.float32, device=device)
    weight = torch.tensor(weights / 2.0, dtype=torch.float32, device=device)
    return alpha, weight


def _scalar_integrated_gradient(
    actual: torch.Tensor,
    function,
    *,
    steps: int,
    batch_size: int = 1,
) -> dict:
    """Fixed-step Gauss--Legendre integrated gradients from zero to ``actual``."""
    baseline = torch.zeros_like(actual)
    with torch.no_grad():
        start_value = float(function(baseline).reshape(-1).sum().detach().cpu())
        end_value = float(function(actual).reshape(-1).sum().detach().cpu())
    nodes, weights = _gauss_legendre_nodes(int(steps), device=str(actual.device))
    average_gradient = torch.zeros_like(actual, dtype=torch.float32)
    for start_index in range(0, int(steps), int(batch_size)):
        alpha = nodes[start_index : start_index + int(batch_size)]
        weight = weights[start_index : start_index + int(batch_size)]
        point = (
            baseline
            + alpha.to(dtype=actual.dtype).view(-1, 1, 1)
            * (actual - baseline)
        ).detach().requires_grad_(True)
        values = function(point).reshape(-1)
        if int(values.numel()) != int(alpha.numel()):
            raise RuntimeError(
                "Batched IG utility must return one scalar per integration point; "
                f"got {tuple(values.shape)} for {int(alpha.numel())} points"
            )
        gradient = torch.autograd.grad(
            (values * weight).sum(), point, retain_graph=False
        )[0]
        average_gradient += gradient.float().sum(dim=0, keepdim=True)
    attribution = (actual - baseline).float() * average_gradient
    estimate = float(attribution.sum().detach().cpu())
    return {
        "gradient": average_gradient.detach(),
        "attribution": attribution.detach(),
        "start_value": start_value,
        "end_value": end_value,
        "expected_delta": end_value - start_value,
        "attributed_delta": estimate,
        "steps": int(steps),
    }


def _make_response_kl_utility(
    runtime: Runtime,
    record: dict,
    *,
    support_only_projection: bool = False,
    support_projection_chunk_size: int = 64,
):
    """Create a callable returning mean response KL reduction as [batch, 1]."""
    prompt_ids = record["prompt_ids"].to(runtime.device, dtype=torch.long)
    response_ids = record["response_ids"].to(runtime.device, dtype=torch.long)
    input_ids = torch.cat([prompt_ids, response_ids]).view(1, -1)
    prompt_len = int(prompt_ids.numel())
    response_len = int(response_ids.numel())
    support_ids = record["support_ids"].to(runtime.device, dtype=torch.long)
    support_mask = record["support_mask"].to(runtime.device, dtype=torch.bool)
    target_logp = record["target_logp"].to(runtime.device).float()
    source_kl = record["source_kl"].to(runtime.device).float()

    def utility(edit: torch.Tensor) -> torch.Tensor:
        batch_size = int(edit.shape[0])
        batched_support_mask = support_mask.unsqueeze(0).expand(batch_size, -1, -1)
        if support_only_projection:
            support_logits = _response_support_logits(
                runtime, input_ids.expand(batch_size, -1),
                prompt_len=prompt_len, response_len=response_len, edit=edit,
                support_ids=support_ids, chunk_size=int(support_projection_chunk_size),
                requires_grad=True,
            )
            logp = conditional_logp_from_support_logits(support_logits, batched_support_mask)
        else:
            logits = _response_logits(
                runtime, input_ids.expand(batch_size, -1),
                prompt_len=prompt_len, response_len=response_len, edit=edit,
                requires_grad=True,
            )
            batched_support_ids = support_ids.unsqueeze(0).expand(batch_size, -1, -1)
            logp, _ = logp_on_support_from_logits(logits, batched_support_ids, batched_support_mask)
        losses = conditional_topk_kl_raw(
            target_logp.unsqueeze(0).expand(batch_size, -1, -1), logp, batched_support_mask
        )
        return torch.stack([
            mean_recovered_utility(source_kl, losses[index])
            for index in range(batch_size)
        ]).unsqueeze(-1)

    return utility


def _attribution_filename(method: str, base: str) -> str:
    return base if str(method) == "ig" else f"{str(method)}_{base}"


def _write_attribution_artifacts(
    output: Path,
    *,
    method: str,
    total_rows: list[dict],
    diagnostics: list[dict],
    primary_mode: str = "teacher_forced",
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    selected = [
        row for row in total_rows
        if str(row.get("mode")) == "teacher_forced"
    ]
    values = [
        float(value)
        for row in selected
        for value in row.get("edit_event_credit", [])
    ]
    mode_summaries = {
        "teacher_forced": {
            "records": len(selected),
            "mean_absolute_token_credit": (
                float(np.mean(np.abs(values))) if values else 0.0
            ),
            "mean_signed_token_credit": float(np.mean(values)) if values else 0.0,
        }
    }
    if str(method) == "ig":
        method_label = "applied_edit_event_scalar_kl_fixed_ig"
    elif str(method) == "relp":
        method_label = "applied_edit_event_scalar_kl_relevance_patching"
    else:
        raise ValueError(f"Unsupported attribution method: {method}")
    write_json(
        output / _attribution_filename(method, "total_response_credit.json"),
        {
            "method": method_label,
            "attribution_method": str(method),
            "credit_unit": "mean conditional KL reduction per response token",
            "interpretation_object": "applied_edit_event",
            "records": total_rows,
            "summaries_by_mode": mode_summaries,
            "primary_interpretation_mode": str(primary_mode),
        },
    )
    write_json(
        output / _attribution_filename(method, "diagnostics.json"),
        {
            "attribution_method": str(method),
            "records": diagnostics,
            "fixed_ig_steps": (
                int(diagnostics[0].get("steps", 0))
                if str(method) == "ig" and diagnostics else None
            ),
        },
    )
    summary = {
        "attribution_method": str(method),
        "total_response_records": len(total_rows),
        "summaries_by_mode": mode_summaries,
        "primary_interpretation_mode": str(primary_mode),
    }
    write_json(output / _attribution_filename(method, "summary.json"), summary)
    return summary


def merge_attribution_shards(
    shard_directories: list[Path],
    output: Path,
    *,
    primary_mode: str = "teacher_forced",
    methods: Iterable[str] = ("ig",),
) -> dict:
    summaries = {}
    support_edges = None
    support_counts = None
    support_positions = 0
    support_minimum = None
    for directory in shard_directories:
        artifact = read_json(directory / "support_mass_distribution.json", {})
        edges = artifact.get("edges", [])
        counts = artifact.get("counts", [])
        if not edges or len(edges) != len(counts) + 1:
            continue
        edges_array = np.asarray(edges, dtype=np.float64)
        counts_array = np.asarray(counts, dtype=np.int64)
        if support_edges is None:
            support_edges = edges_array
            support_counts = np.zeros_like(counts_array)
        elif not np.array_equal(support_edges, edges_array):
            raise RuntimeError("Attribution shards used incompatible support-mass histogram bins")
        support_counts += counts_array
        support_positions += int(artifact.get("positions", int(counts_array.sum())))
        shard_minimum = artifact.get("minimum")
        if shard_minimum is not None and math.isfinite(float(shard_minimum)):
            support_minimum = (
                float(shard_minimum)
                if support_minimum is None
                else min(float(support_minimum), float(shard_minimum))
            )
    if support_edges is not None and support_counts is not None:
        write_json(
            output / "support_mass_distribution.json",
            {
                "population": "teacher_forced_response_positions_min_source_target",
                "support_mass_definition": "minimum_source_target_conditional_support_mass",
                "positions": int(support_positions),
                "counts": support_counts.tolist(),
                "edges": support_edges.tolist(),
                "minimum": support_minimum,
            },
        )

    for method in methods:
        collections = {
            "total_response_credit.json": [],
            "diagnostics.json": [],
        }
        for directory in shard_directories:
            for filename, destination in collections.items():
                artifact = read_json(
                    directory / _attribution_filename(str(method), filename), {}
                )
                destination.extend(artifact.get("records", []))
        summaries[str(method)] = _write_attribution_artifacts(
            output,
            method=str(method),
            total_rows=collections["total_response_credit.json"],
            diagnostics=collections["diagnostics.json"],
            primary_mode=primary_mode,
        )
    return summaries


def prepare_attribution_record_shards(
    config: dict,
    record_directories: dict[str, Path],
    *,
    shard_count: int,
) -> list[dict[str, list[Path]]]:
    """Plan stage-2 record shards using lightweight JSON sidecars.

    The parent process balances sequence lengths and returns paths for workers
    to load one at a time, avoiding a full activation preload in each worker.
    """
    settings = config["attribution"]
    max_records = int(settings.get("max_records", 0))
    enabled_modes = set(str(mode) for mode in settings.get("modes", record_directories))
    count = max(1, int(shard_count))
    plans: list[dict[str, list[Path]]] = [dict() for _ in range(count)]

    for mode, directory in record_directories.items():
        if mode not in enabled_modes or not directory.exists():
            continue
        # Apply the existing maximum/direction/source selection semantics once,
        # but do not shard here: the parent owns the one global balancing pass.
        paths = _record_paths(
            directory,
            maximum=max_records,
            shard_index=0,
            shard_count=1,
        )
        weighted: list[tuple[int, Path]] = []
        for path in tqdm(
            paths,
            desc=f"index attribution records ({mode})",
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
            plans[0][mode] = [
                path for _, path in sorted(weighted, key=lambda item: (item[0], item[1].name))
            ]
            continue

        shard_weighted: list[list[tuple[int, Path]]] = [[] for _ in range(count)]
        shard_loads = [0 for _ in range(count)]
        for cost, path in sorted(weighted, key=lambda item: (-item[0], item[1].name)):
            destination = min(
                range(count),
                key=lambda index: (shard_loads[index], index),
            )
            shard_weighted[destination].append((int(cost), path))
            shard_loads[destination] += int(cost)
        for index in range(count):
            # Use short-to-long processing order within each
            # balanced shard without deserializing records just to learn length.
            plans[index][mode] = [
                path
                for _, path in sorted(
                    shard_weighted[index], key=lambda item: (item[0], item[1].name)
                )
            ]
    return plans


def compute_scalar_kl_attribution(
    config: dict,
    runtime: Runtime,
    record_directories: dict[str, Path],
    output: Path,
    *,
    shard_index: int = 0,
    shard_count: int = 1,
    record_paths_by_mode: dict[str, list[Path]] | None = None,
) -> dict:
    """Compute fixed-step IG attribution on applied edit events.

    Full residual-space sensitivities are never serialized into JSON.
    """
    settings = config["attribution"]
    methods = [str(method) for method in settings.get("methods", ["ig"])]
    ig_steps = int(settings.get("ig_steps", 16))
    integration_batch_size = int(settings.get("integration_batch_size", 1))
    max_records = int(settings.get("max_records", 0))
    support_only_projection = bool(settings.get("support_only_projection", False))
    support_projection_chunk_size = int(settings.get("support_projection_chunk_size", 64))
    enabled_modes = set(str(mode) for mode in settings.get("modes", record_directories))
    collections = {
        method: {
            "total": [], "diagnostics": []
        }
        for method in methods
    }
    # Support mass is already resident while stage 3 streams teacher-forced
    # records.  Accumulate a compact token-level histogram here so the report
    # can show the true distribution without reopening every large tensor trace.
    support_mass_edges = np.linspace(0.0, 1.0, 51, dtype=np.float64)
    support_mass_counts = np.zeros(len(support_mass_edges) - 1, dtype=np.int64)
    support_mass_positions = 0
    support_mass_minimum = None
    active_integration_batch_size = max(1, min(int(integration_batch_size), int(ig_steps)))
    for mode, directory in record_directories.items():
        if mode not in enabled_modes or not directory.exists():
            continue
        if record_paths_by_mode is None:
            local_plan = prepare_attribution_record_shards(
                config,
                {mode: directory},
                shard_count=int(shard_count),
            )
            paths = local_plan[int(shard_index)].get(mode, [])
        else:
            paths = list(record_paths_by_mode.get(mode, []))
        progress = tqdm(
            paths,
            desc=(
                f"applied-edit attribution ({mode})"
                if int(shard_count) == 1
                else f"applied-edit attribution ({mode}, worker {int(shard_index) + 1})"
            ),
            unit="record",
        )
        for path in progress:
            # Stream one tensor-heavy trace at a time.  Once this iteration is
            # complete, the record can be released instead of keeping the full
            # worker shard resident in host RAM.
            record = torch.load(path, map_location="cpu")
            if mode == "teacher_forced":
                support_mass = record.get("support_mass")
                if support_mass is not None:
                    values = support_mass.detach().float().cpu().numpy().reshape(-1)
                    values = values[np.isfinite(values)]
                    if values.size:
                        counts, _ = np.histogram(
                            np.clip(values.astype(np.float64, copy=False), 0.0, 1.0),
                            bins=support_mass_edges,
                        )
                        support_mass_counts += counts.astype(np.int64, copy=False)
                        support_mass_positions += int(values.size)
                        local_minimum = float(values.min())
                        support_mass_minimum = (
                            local_minimum
                            if support_mass_minimum is None
                            else min(float(support_mass_minimum), local_minimum)
                        )
            edit_actual = record["edit"].to(runtime.device, dtype=torch.float32)
            utility = _make_response_kl_utility(
                runtime, record,
                support_only_projection=support_only_projection,
                support_projection_chunk_size=support_projection_chunk_size,
            )
            method_results = {}
            if "ig" in methods:
                result, used_batch = _adaptive_cuda_call(
                    lambda batch: _scalar_integrated_gradient(
                        edit_actual, lambda edit: utility(edit)[:, 0],
                        steps=ig_steps, batch_size=batch,
                    ),
                    batch_size=active_integration_batch_size,
                    device=runtime.device, label=f"applied-edit IG {mode}",
                )
                active_integration_batch_size = min(active_integration_batch_size, int(used_batch))
                method_results["ig"] = [result]
            if "relp" in methods:
                method_results["relp"] = multi_scalar_relevance_patching(
                    edit_actual, utility, utility_count=1, runtime=runtime,
                )

            for method in methods:
                results = method_results.get(method)
                if results is None:
                    continue
                for result in results:
                    edit_attribution = result["attribution"].detach().float().cpu()
                    if edit_attribution.ndim == 3:
                        edit_attribution = edit_attribution[0]
                    source_by_token = edit_attribution.sum(dim=-1)
                    common = {
                        "attribution_method": method,
                        "record_id": str(record["record_id"]),
                        "source_id": str(record["source_id"]),
                        "response_id": str(record["response_id"]),
                        "direction": str(record["direction"]),
                        "mode": mode,
                        "kind": "total",
                        "edit_event_credit": source_by_token.tolist(),
                        "attributed_delta": float(result["attributed_delta"]),
                    }
                    common.update(
                        {
                            "no_edit_utility": float(result["start_value"]),
                            "expected_delta": float(result["expected_delta"]),
                        }
                    )
                    if "steps" in result:
                        common["steps"] = int(result["steps"])
                    if method == "relp":
                        common["relp_rules"] = list(result.get("relp_rules", []))
                        common["relp_attention_rule"] = str(result.get("relp_attention_rule", "none"))
                    collections[method]["total"].append(common)
                    diagnostic = {
                        "record_id": str(record["record_id"]),
                        "mode": mode,
                        "kind": "total",
                        "attribution_method": method,
                    }
                    if "steps" in result:
                        diagnostic["steps"] = int(result["steps"])
                    if method == "relp":
                        diagnostic.update(
                            {
                                "relp_rules": list(result.get("relp_rules", [])),
                                "relp_attention_rule": str(result.get("relp_attention_rule", "none")),
                                "relp_downstream_layers": int(result.get("relp_downstream_layers", 0)),
                                "relp_rmsnorm_modules": int(result.get("relp_rmsnorm_modules", 0)),
                                "relp_gated_mlp_modules": int(result.get("relp_gated_mlp_modules", 0)),
                                "relp_per_layer_input_gates": int(result.get("relp_per_layer_input_gates", 0)),
                            }
                        )
                    collections[method]["diagnostics"].append(diagnostic)

    write_json(
        output / "support_mass_distribution.json",
        {
            "population": "teacher_forced_response_positions_min_source_target",
            "support_mass_definition": "minimum_source_target_conditional_support_mass",
            "positions": int(support_mass_positions),
            "counts": support_mass_counts.tolist(),
            "edges": support_mass_edges.tolist(),
            "minimum": support_mass_minimum,
        },
    )
    summaries = {}
    for method in methods:
        rows = collections[method]
        summaries[method] = _write_attribution_artifacts(
            output,
            method=method,
            total_rows=rows["total"],
            diagnostics=rows["diagnostics"],
            primary_mode=str(settings["primary_interpretation_mode"]),
        )
    return {
        "methods": methods,
        "summaries": summaries,
        "total_records": {method: len(collections[method]["total"]) for method in methods},
    }


def qualitative_record_key(row: dict) -> tuple[str, str, str]:
    """Stable conceptual record key independent of artifact storage details."""
    return (
        str(row.get("source_id", "")),
        str(row.get("response_id", "")),
        str(row.get("direction", "")),
    )


def select_qualitative_record_keys(
    rows: Iterable[dict],
    *,
    fraction: float,
    seed: int,
) -> set[tuple[str, str, str]]:
    """Select an exact deterministic fraction of conceptual records by hash rank."""
    unique = sorted({qualitative_record_key(row) for row in rows})
    if not unique:
        return set()
    count = max(1, int(math.ceil(float(fraction) * len(unique))))
    ranked = sorted(
        unique,
        key=lambda key: hashlib.sha256(
            f"qualitative:{int(seed)}:{key[0]}:{key[1]}:{key[2]}".encode("utf-8")
        ).digest(),
    )
    return set(ranked[:count])
