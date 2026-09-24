#!/usr/bin/env python3
from __future__ import annotations

"""Build top-k bucketed teacher distributions for off-policy KL edit distillation."""

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import shutil
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from behavior_pipeline.runtime_env import configure_runtime_caches  # noqa: E402

configure_runtime_caches(Path(__file__).resolve().parents[1])

import numpy as np
import torch
import torch.nn.functional as F

from behavior_pipeline.model_utils import load_causal_lm  # noqa: E402
from behavior_pipeline.artifact_identity import file_sha256  # noqa: E402
from behavior_pipeline.rollouts import (  # noqa: E402
    load_grouped_rollout_splits,
)


def _requested_cache_config(args: argparse.Namespace) -> dict:
    generation = _generation_provenance(Path(args.data_dir))
    return {
        "cache_format": "topk_kl_distill",
        "model_name": str(args.model_name),
        "data_dir": str(args.data_dir),
        "requested_splits": ["train", "val", "test"],
        "top_k": int(args.top_k),
        "prefix_policy": str(args.prefix_policy),
        "max_response_tokens": int(args.max_response_tokens),
        "shard_size": int(args.shard_size),
        "logp_dtype": str(args.logp_dtype),
        "model_dtype": str(args.dtype),
        "support_mode": "union_control_intervention_topk_plus_actual",
        "token_id_policy": "saved_generation_ids_required",
        "seed": int(args.seed),
        "generation_metadata_sha256": generation["sha256"],
        "generation_top_k": generation["generation_top_k"],
        "generation_and_distillation_top_k_are_independent": True,
        "generation": generation["metadata"],
    }


def _generation_provenance(data_dir: Path) -> dict:
    path = data_dir / "generation_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing generation provenance: {path}; rerun response generation"
        )
    payload = path.read_bytes()
    metadata = json.loads(payload.decode("utf-8"))
    # Generation truncation and distillation support are separate policies.
    # In particular, top_k=-1 generation may still use a finite conditional
    # support for the source/target KL objective.
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "metadata": metadata,
        "generation_top_k": int(metadata.get("top_k", -1)),
    }


def _validate_existing_cache(cache_dir: Path, args: argparse.Namespace) -> bool:
    meta_path = cache_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = _requested_cache_config(args)
    mismatches = []
    for key, value in expected.items():
        actual = metadata.get(key)
        if actual != value:
            mismatches.append({"key": key, "expected": value, "actual": metadata.get(key)})
    split_meta = metadata.get("splits")
    if not isinstance(split_meta, dict):
        mismatches.append({"key": "splits", "expected": "dict", "actual": type(split_meta).__name__})
    else:
        for split in expected["requested_splits"]:
            if split not in split_meta:
                mismatches.append({"key": f"splits.{split}", "expected": "present", "actual": "missing"})
                continue
            for shard in split_meta[split].get("shards", []):
                shard_path = cache_dir / shard.get("path", "")
                if not shard_path.exists():
                    mismatches.append({"key": f"{split}.shard", "expected": str(shard_path), "actual": "missing"})
                    continue
                try:
                    obj = torch.load(shard_path, map_location="cpu")
                    num_rows = len(obj.get("rows", []))
                except Exception as exc:
                    mismatches.append({"key": f"{split}.shard", "expected": str(shard_path), "actual": f"unreadable: {exc}"})
                    continue
                expected_rows = int(shard.get("num_sequences", -1))
                if num_rows != expected_rows:
                    mismatches.append({"key": f"{split}.shard_rows", "expected": expected_rows, "actual": num_rows})
                expected_sha = shard.get("sha256")
                actual_sha = file_sha256(shard_path)
                if expected_sha != actual_sha:
                    mismatches.append({"key": f"{split}.shard_sha256", "expected": expected_sha, "actual": actual_sha})
    if mismatches:
        print(json.dumps({"topk_kl_cache": "reuse_rejected", "cache_dir": str(cache_dir), "mismatches": mismatches}, default=str), flush=True)
        return False
    return True


def _validate_data_dir(data_dir: Path, splits: list[str]) -> None:
    missing = [str(data_dir / f"{split}.jsonl") for split in splits if not (data_dir / f"{split}.jsonl").exists()]
    if missing:
        raise SystemExit(
            "Missing generated rollout split files before cache build:\n"
            + "\n".join(f"  {path}" for path in missing)
            + "\nSet DATA_DIR/--data_dir to the directory produced by scripts/generate_responses.py, "
            "or run response generation first."
        )


def _saved_ids(row: dict, key: str) -> list[int] | None:
    values = row.get(key)
    if values is None:
        return None
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{key} must be a list of token IDs when present")
    return [int(value) for value in values]


def _response_rows(
    source_items, *, prefix_policy: str, max_response_tokens: int
) -> list[dict]:
    rows = []
    for source_id, rec in list(source_items):
        controls = list(rec["control"])
        interventions = list(rec["intervention"])
        if not controls or not interventions:
            continue
        control_prompt_ids = _saved_ids(controls[0], "prompt_input_ids")
        intervention_prompt_ids = _saved_ids(
            interventions[0], "prompt_input_ids"
        )
        if control_prompt_ids is None or intervention_prompt_ids is None:
            raise ValueError(
                f"source {source_id!r} is missing saved prompt_input_ids; "
                "rerun scripts/generate_responses.py"
            )
        arm_rows = []
        if prefix_policy in {"both_arms", "control"}:
            arm_rows.extend(("control", idx, row) for idx, row in enumerate(controls))
        if prefix_policy in {"both_arms", "intervention"}:
            arm_rows.extend(("intervention", idx, row) for idx, row in enumerate(interventions))
        for response_arm, response_idx, row in arm_rows:
            response_ids = _saved_ids(row, "response_input_ids")
            if response_ids is None:
                raise ValueError(
                    f"source {source_id!r} response {response_arm}:{response_idx} "
                    "is missing saved response_input_ids; rerun "
                    "scripts/generate_responses.py"
                )
            original_response_length = len(response_ids)
            if int(max_response_tokens) > 0:
                response_ids = response_ids[: int(max_response_tokens)]
            if not response_ids:
                continue
            rows.append(
                {
                    "source_id": str(source_id),
                    "response_id": (
                        f"{source_id}:{response_arm}:"
                        f"{int(row.get('rollout_id', response_idx))}"
                    ),
                    "response_arm": str(response_arm),
                    "rollout_id": int(row.get("rollout_id", response_idx)),
                    "response_token_source": "saved_generation_ids",
                    "finish_reason": row.get("finish_reason"),
                    "response_truncated": bool(row.get("response_truncated", False)) or len(response_ids) < original_response_length,
                    "control_prompt_ids": torch.tensor(control_prompt_ids, dtype=torch.long),
                    "intervention_prompt_ids": torch.tensor(intervention_prompt_ids, dtype=torch.long),
                    "response_ids": torch.tensor(response_ids, dtype=torch.long),
                }
            )
    return rows


def _response_logprobs(model, prompt_ids: torch.Tensor, response_ids: torch.Tensor, *, device: str) -> torch.Tensor:
    input_ids = torch.cat([prompt_ids, response_ids]).to(device=device, dtype=torch.long).view(1, -1)
    prompt_len = int(prompt_ids.numel())
    response_len = int(response_ids.numel())
    logits = model(input_ids=input_ids, use_cache=False).logits[:, prompt_len - 1 : prompt_len - 1 + response_len, :]
    return F.log_softmax(logits[0].float(), dim=-1)


def _build_cache_row(model, row: dict, *, top_k: int, logp_dtype: str, device: str) -> dict:
    with torch.no_grad():
        logp_control = _response_logprobs(model, row["control_prompt_ids"], row["response_ids"], device=device)
        logp_intervention = _response_logprobs(model, row["intervention_prompt_ids"], row["response_ids"], device=device)
    response_ids = row["response_ids"].cpu()
    response_len = int(response_ids.numel())
    topk = min(int(top_k), int(logp_control.shape[-1]))
    out_support = []
    out_mask = []
    out_logp_control = []
    out_logp_intervention = []
    out_logp_control_other = []
    out_logp_intervention_other = []
    max_support = 0
    for pos in range(response_len):
        ids = torch.cat(
            [
                torch.topk(logp_control[pos], k=topk).indices.cpu(),
                torch.topk(logp_intervention[pos], k=topk).indices.cpu(),
                response_ids[pos : pos + 1],
            ]
        )
        support = torch.unique(ids, sorted=True).to(dtype=torch.long)
        max_support = max(max_support, int(support.numel()))
        out_support.append(support)
    for pos, support in enumerate(out_support):
        control_vals = logp_control[pos, support.to(device=logp_control.device)].detach().cpu()
        intervention_vals = logp_intervention[pos, support.to(device=logp_intervention.device)].detach().cpu()
        control_mass = control_vals.float().exp().sum().clamp(max=1.0)
        intervention_mass = intervention_vals.float().exp().sum().clamp(max=1.0)
        out_logp_control_other.append(torch.log1p((-control_mass).clamp(min=-1.0, max=0.0)).clamp_min(math.log(1e-12)))
        out_logp_intervention_other.append(torch.log1p((-intervention_mass).clamp(min=-1.0, max=0.0)).clamp_min(math.log(1e-12)))
        pad = max_support - int(support.numel())
        out_mask.append(torch.cat([torch.ones(int(support.numel()), dtype=torch.bool), torch.zeros(pad, dtype=torch.bool)]))
        out_logp_control.append(torch.cat([control_vals, torch.zeros(pad, dtype=control_vals.dtype)]))
        out_logp_intervention.append(torch.cat([intervention_vals, torch.zeros(pad, dtype=intervention_vals.dtype)]))
        if pad:
            support = torch.cat([support, torch.zeros(pad, dtype=torch.long)])
        out_support[pos] = support
    dtype = torch.float16 if str(logp_dtype) == "fp16" else torch.float32
    return {
        **row,
        "support_ids": torch.stack(out_support, dim=0).to(dtype=torch.int32),
        "support_mask": torch.stack(out_mask, dim=0),
        "logp_control": torch.stack(out_logp_control, dim=0).to(dtype=dtype),
        "logp_intervention": torch.stack(out_logp_intervention, dim=0).to(dtype=dtype),
        "logp_control_other": torch.stack(out_logp_control_other, dim=0).to(dtype=torch.float32),
        "logp_intervention_other": torch.stack(out_logp_intervention_other, dim=0).to(dtype=torch.float32),
    }


def _parse_devices(devices: list[str] | None, device: str | None = None) -> list[str]:
    """Resolve space- or comma-separated device arguments into unique devices."""
    values: list[str] = []
    if devices:
        for value in devices:
            values.extend(part.strip() for part in str(value).split(",") if part.strip())
    elif device:
        values.extend(part.strip() for part in str(device).split(",") if part.strip())
    else:
        values = ["cuda:0"]
    resolved: list[str] = []
    for value in values:
        if value not in resolved:
            resolved.append(value)
    if not resolved:
        raise ValueError("at least one device is required")
    return resolved


def _contiguous_ranges(num_items: int, num_workers: int) -> list[tuple[int, int]]:
    """Partition an ordered source list into deterministic contiguous ranges."""
    num_items = int(num_items)
    num_workers = max(1, int(num_workers))
    base, extra = divmod(num_items, num_workers)
    ranges = []
    start = 0
    for rank in range(num_workers):
        size = base + (1 if rank < extra else 0)
        ranges.append((start, start + size))
        start += size
    return ranges


def _write_shard(cache_dir: Path, split: str, shard_idx: int, rows: list[dict]) -> dict:
    path = f"{split}_shard_{int(shard_idx):04d}.pt"
    shard_path = cache_dir / path
    torch.save({"rows": rows}, shard_path)
    return {
        "path": path,
        "num_sequences": int(len(rows)),
        "sha256": file_sha256(shard_path),
    }


def _worker_build_cache(
    *,
    rank: int,
    device: str,
    worker_count: int,
    data_dir: str,
    temp_dir: str,
    model_name: str,
    dtype: str,
    top_k: int,
    logp_dtype: str,
    prefix_policy: str,
    max_response_tokens: int,
    shard_size: int,
    requested_splits: list[str],
    source_ranges: dict[str, tuple[int, int]],
    result_path: str,
) -> None:
    """Build temporary cache chunks for one contiguous source partition."""
    try:
        if str(device).startswith("cuda:") and torch.cuda.is_available():
            torch.cuda.set_device(int(str(device).split(":", 1)[1]))
        loaded = load_causal_lm(str(model_name), device=str(device), dtype=str(dtype))
        model = loaded.model
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        grouped = load_grouped_rollout_splits(Path(data_dir), requested_splits)
        worker_dir = Path(temp_dir) / f"worker_{int(rank):02d}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        summary: dict[str, dict] = {}
        for split in requested_splits:
            all_sources = list(grouped[split].items())
            start, end = source_ranges[split]
            source_items = all_sources[int(start) : int(end)]
            raw_rows = _response_rows(
                source_items,
                prefix_policy=str(prefix_policy),
                max_response_tokens=int(max_response_tokens),
            )
            chunk_paths: list[str] = []
            chunk_rows: list[dict] = []
            chunk_idx = 0
            for idx, row in enumerate(raw_rows):
                chunk_rows.append(
                    _build_cache_row(
                        model,
                        row,
                        top_k=int(top_k),
                        logp_dtype=str(logp_dtype),
                        device=str(device),
                    )
                )
                if len(chunk_rows) >= max(1, int(shard_size)):
                    chunk_path = worker_dir / f"{split}_chunk_{chunk_idx:04d}.pt"
                    torch.save({"rows": chunk_rows}, chunk_path)
                    chunk_paths.append(str(chunk_path))
                    chunk_rows = []
                    chunk_idx += 1
                if (idx + 1) % 25 == 0 or (idx + 1) == len(raw_rows):
                    print(
                        json.dumps(
                            {
                                "topk_kl_cache_worker": int(rank),
                                "device": str(device),
                                "split": split,
                                "done": idx + 1,
                                "total": len(raw_rows),
                            }
                        ),
                        flush=True,
                    )
            if chunk_rows:
                chunk_path = worker_dir / f"{split}_chunk_{chunk_idx:04d}.pt"
                torch.save({"rows": chunk_rows}, chunk_path)
                chunk_paths.append(str(chunk_path))
            summary[split] = {
                "source_start": int(start),
                "source_end": int(end),
                "sources": int(len(source_items)),
                "sequences": int(len(raw_rows)),
                "chunks": chunk_paths,
            }
        Path(result_path).write_text(
            json.dumps({"rank": int(rank), "ok": True, "splits": summary}, indent=2),
            encoding="utf-8",
        )
    except BaseException as exc:
        Path(result_path).parent.mkdir(parents=True, exist_ok=True)
        Path(result_path).write_text(
            json.dumps(
                {
                    "rank": int(rank),
                    "ok": False,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise


def _build_workers(
    *,
    args: argparse.Namespace,
    devices: list[str],
    grouped: dict,
    requested_splits: list[str],
    temp_dir: Path,
) -> list[dict]:
    """Run one model replica per device and return ordered worker summaries."""
    max_sources = max(len(grouped[split]) for split in requested_splits)
    worker_count = min(len(devices), max(1, int(max_sources)))
    devices = devices[:worker_count]
    ranges_by_split = {
        split: _contiguous_ranges(len(grouped[split]), worker_count)
        for split in requested_splits
    }
    print(
        json.dumps(
            {
                "topk_kl_cache": "parallel_build",
                "workers": worker_count,
                "devices": devices,
                "source_partition": "contiguous",
            }
        ),
        flush=True,
    )

    worker_kwargs = []
    for rank, device in enumerate(devices):
        worker_kwargs.append(
            dict(
                rank=rank,
                device=device,
                worker_count=worker_count,
                data_dir=str(args.data_dir),
                temp_dir=str(temp_dir),
                model_name=str(args.model_name),
                dtype=str(args.dtype),
                top_k=int(args.top_k),
                logp_dtype=str(args.logp_dtype),
                prefix_policy=str(args.prefix_policy),
                max_response_tokens=int(args.max_response_tokens),
                shard_size=int(args.shard_size),
                requested_splits=requested_splits,
                source_ranges={
                    split: ranges_by_split[split][rank]
                    for split in requested_splits
                },
                result_path=str(temp_dir / f"worker_{rank:02d}" / "summary.json"),
            )
        )

    if worker_count == 1:
        _worker_build_cache(**worker_kwargs[0])
    else:
        ctx = mp.get_context("spawn")
        processes = [
            ctx.Process(
                target=_worker_build_cache,
                kwargs=kwargs,
                name=f"topk-cache-{rank}",
            )
            for rank, kwargs in enumerate(worker_kwargs)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join()

    results = []
    failures = []
    for rank, kwargs in enumerate(worker_kwargs):
        result_path = Path(kwargs["result_path"])
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            results.append(result)
            if not result.get("ok"):
                failures.append(result)
        else:
            exitcode = None if worker_count == 1 else processes[rank].exitcode
            failures.append(
                {
                    "rank": rank,
                    "ok": False,
                    "error": f"worker produced no summary (exitcode={exitcode})",
                }
            )
    if failures:
        details = "\n\n".join(
            result.get("traceback") or result.get("error", "unknown worker failure")
            for result in failures
        )
        raise RuntimeError(f"top-k cache worker failure:\n{details}")
    return sorted(results, key=lambda result: int(result["rank"]))


def _merge_worker_chunks(
    *,
    cache_dir: Path,
    split: str,
    worker_results: list[dict],
    shard_size: int,
) -> dict:
    """Merge temporary worker chunks into deterministic final cache shards."""
    final_shards: list[dict] = []
    current_rows: list[dict] = []
    response_token_sources = {"saved_generation_ids": 0}
    total_sequences = 0
    shard_idx = 0

    def flush() -> None:
        nonlocal current_rows, shard_idx
        if not current_rows:
            return
        final_shards.append(_write_shard(cache_dir, split, shard_idx, current_rows))
        current_rows = []
        shard_idx += 1

    for result in worker_results:
        split_summary = result["splits"][split]
        total_sequences += int(split_summary["sequences"])
        for chunk_path in split_summary["chunks"]:
            obj = torch.load(chunk_path, map_location="cpu")
            for row in obj.get("rows", []):
                response_token_sources["saved_generation_ids"] += int(
                    row.get("response_token_source") == "saved_generation_ids"
                )
                current_rows.append(row)
                if len(current_rows) >= max(1, int(shard_size)):
                    flush()
    flush()
    if total_sequences == 0 or not final_shards:
        raise RuntimeError(f"split {split!r} produced no cache rows")
    return {
        "sequences": int(total_sequences),
        "response_token_sources": response_token_sources,
        "shards": final_shards,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--model_name", default="google/gemma-4-E4B-it")
    ap.add_argument("--top_k", type=int, default=64)
    ap.add_argument("--prefix_policy", choices=["both_arms", "control", "intervention"], default="both_arms")
    ap.add_argument("--max_response_tokens", type=int, default=32768)
    ap.add_argument("--shard_size", type=int, default=128)
    ap.add_argument("--logp_dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="One or more devices, e.g. --devices cuda:0 cuda:1. Comma-separated values are also accepted.",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="Single-device alias for --devices.",
    )
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    devices = _parse_devices(args.devices, args.device)
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    cache_dir = Path(args.cache_dir)
    meta_path = cache_dir / "metadata.json"
    if meta_path.exists() and _validate_existing_cache(cache_dir, args):
        print(json.dumps({"topk_kl_cache": "reuse", "cache_dir": str(cache_dir)}), flush=True)
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    if meta_path.exists():
        meta_path.unlink()
    requested_splits = ["train", "val", "test"]
    _validate_data_dir(Path(args.data_dir), requested_splits)
    grouped = load_grouped_rollout_splits(Path(args.data_dir), requested_splits)
    for split in requested_splits:
        source_items = list(grouped[split].items())
        raw_rows = _response_rows(
            source_items,
            prefix_policy=str(args.prefix_policy),
            max_response_tokens=int(args.max_response_tokens),
        )
        if not source_items or not raw_rows:
            raise RuntimeError(
                f"split {split!r} has no balanced rollout records; "
                "the train/val/test contract requires every split to be non-empty"
            )

    for split in requested_splits:
        for stale in cache_dir.glob(f"{split}_shard_*.pt"):
            stale.unlink()
    temp_dir = cache_dir / ".parallel_build"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        worker_results = _build_workers(
            args=args,
            devices=devices,
            grouped=grouped,
            requested_splits=requested_splits,
            temp_dir=temp_dir,
        )
        split_meta = {}
        for split in requested_splits:
            merged = _merge_worker_chunks(
                cache_dir=cache_dir,
                split=split,
                worker_results=worker_results,
                shard_size=int(args.shard_size),
            )
            merged["sources"] = int(len(grouped[split]))
            split_meta[split] = merged
            print(
                json.dumps(
                    {
                        "topk_kl_cache_split": split,
                        "sources": merged["sources"],
                        "sequences": merged["sequences"],
                        "shards": len(merged["shards"]),
                    }
                ),
                flush=True,
            )
        metadata = {
            **_requested_cache_config(args),
            "build_parallelism": {
                "workers": int(min(len(devices), max(len(grouped[s]) for s in requested_splits))),
                "devices": devices,
                "partition": "contiguous_sources",
            },
            "splits": split_meta,
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps({"topk_kl_cache": "built", "cache_dir": str(cache_dir), "splits": split_meta}), flush=True)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
