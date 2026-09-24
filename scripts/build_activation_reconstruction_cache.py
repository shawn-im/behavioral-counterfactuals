#!/usr/bin/env python3
from __future__ import annotations

"""Cache paired residual activations for normalized-MSE editor training.

The source population is the existing top-k KL cache, so train/val/test rows,
response truncation, and fixed response prefixes are exactly shared between the
behavior-reconstruction and activation-reconstruction experiments.
"""

import argparse
import gc
import json
import multiprocessing as mp
import shutil
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from behavior_pipeline.runtime_env import configure_runtime_caches  # noqa: E402
configure_runtime_caches(Path(__file__).resolve().parents[1])

import torch  # noqa: E402

from behavior_pipeline.activation_reconstruction import ACTIVATION_CACHE_FORMAT
from behavior_pipeline.artifact_identity import cache_identity, file_sha256  # noqa: E402
from behavior_pipeline.model_utils import load_causal_lm, resolve_hidden_size  # noqa: E402
from behavior_pipeline.tasks import FINAL_QUESTION_PREFIX  # noqa: E402
from behavior_pipeline.topk_kl import (  # noqa: E402
    FINAL_QUESTION_RESPONSE_POSITION_POLICY,
    final_question_start_index,
)


class _StopAfterCapture(RuntimeError):
    pass


def _parse_devices(values: list[str] | None, fallback: str | None) -> list[str]:
    parts: list[str] = []
    for value in values or ([fallback] if fallback else ["cuda:0"]):
        if value is None:
            continue
        parts.extend(piece.strip() for piece in str(value).split(",") if piece.strip())
    out: list[str] = []
    for part in parts:
        if part not in out:
            out.append(part)
    if not out:
        raise ValueError("at least one device is required")
    return out


def _light_row(row: dict, index: int) -> dict:
    return {
        "row_index": int(index),
        "source_id": str(row["source_id"]),
        "response_id": str(row["response_id"]),
        "response_arm": str(row.get("response_arm", "")),
        "rollout_id": int(row.get("rollout_id", -1)),
        "control_prompt_ids": row["control_prompt_ids"].to(dtype=torch.long).cpu(),
        "intervention_prompt_ids": row["intervention_prompt_ids"].to(dtype=torch.long).cpu(),
        "response_ids": row["response_ids"].to(dtype=torch.long).cpu(),
    }


def _contiguous_source_partitions(rows: list[dict], workers: int) -> list[list[dict]]:
    groups: list[list[dict]] = []
    current: list[dict] = []
    current_source = None
    for row in rows:
        source = str(row["source_id"])
        if current and source != current_source:
            groups.append(current)
            current = []
        current.append(row)
        current_source = source
    if current:
        groups.append(current)
    if not groups:
        return [[] for _ in range(workers)]
    total = sum(len(group) for group in groups)
    target = max(1.0, total / float(workers))
    partitions: list[list[dict]] = [[] for _ in range(workers)]
    worker = 0
    count = 0
    for group in groups:
        if worker < workers - 1 and partitions[worker] and count + len(group) > target:
            worker += 1
            count = 0
        partitions[worker].extend(group)
        count += len(group)
    return partitions


def _source_split_metadata(source_root: Path, split: str) -> tuple[dict, list[dict]]:
    meta = json.loads((Path(source_root) / "metadata.json").read_text(encoding="utf-8"))
    if (
        meta.get("cache_format") != "topk_kl_distill"
    ):
        raise ValueError(
            "activation reconstruction requires a topk_kl_distill cache"
        )
    split_meta = meta.get("splits", {}).get(str(split))
    if not isinstance(split_meta, dict):
        raise ValueError(f"source cache metadata has no split {split!r}")
    shards = list(split_meta.get("shards", []))
    return meta, shards


def _source_index_partitions(
    source_root: Path, split: str, workers: int
) -> list[list[int]]:
    """Partition source-cache rows without retaining their tensors in the parent."""
    _, shards = _source_split_metadata(source_root, split)
    lightweight: list[dict] = []
    row_index = 0
    for shard in shards:
        obj = torch.load(Path(source_root) / str(shard["path"]), map_location="cpu")
        for row in obj.get("rows", []):
            lightweight.append(
                {"source_id": str(row["source_id"]), "row_index": int(row_index)}
            )
            row_index += 1
        del obj
    return [
        [int(row["row_index"]) for row in partition]
        for partition in _contiguous_source_partitions(lightweight, workers)
    ]


def _load_light_rows_by_index(
    source_root: Path, split: str, row_indices: list[int]
) -> list[dict]:
    """Load only the prompt/response tensors assigned to one worker.

    The worker receives integer indices through multiprocessing and reconstructs
    the tensor-rich rows locally from disk. Shards outside the worker's contiguous
    index range are skipped without loading.
    """
    if not row_indices:
        return []
    wanted = {int(index) for index in row_indices}
    first = min(wanted)
    last = max(wanted)
    _, shards = _source_split_metadata(source_root, split)
    selected: dict[int, dict] = {}
    offset = 0
    for shard in shards:
        count = int(shard.get("num_sequences", 0))
        shard_start = offset
        shard_stop = offset + count
        offset = shard_stop
        if shard_stop <= first:
            continue
        if shard_start > last:
            break
        obj = torch.load(Path(source_root) / str(shard["path"]), map_location="cpu")
        rows = obj.get("rows", [])
        for local_index, row in enumerate(rows):
            global_index = shard_start + int(local_index)
            if global_index in wanted:
                selected[global_index] = _light_row(row, global_index)
        del obj
    missing = [index for index in row_indices if int(index) not in selected]
    if missing:
        raise RuntimeError(
            f"source cache split {split!r} is missing assigned row indices: {missing[:10]}"
        )
    return [selected[int(index)] for index in row_indices]


def _capture_batch(
    model,
    layer,
    rows: list[dict],
    *,
    tokenizer,
    device: str,
    pad_token_id: int,
    storage_dtype: torch.dtype,
) -> list[dict]:
    sequences: list[torch.Tensor] = []
    specs: list[tuple[int, str, int, int, int]] = []
    for row_idx, row in enumerate(rows):
        response = row["response_ids"]
        for arm, prompt_key in (("control", "control_prompt_ids"), ("intervention", "intervention_prompt_ids")):
            prompt = row[prompt_key]
            sequences.append(torch.cat([prompt, response]).to(dtype=torch.long))
            question_start = final_question_start_index(tokenizer, prompt)
            specs.append(
                (
                    row_idx,
                    arm,
                    question_start,
                    int(prompt.numel()),
                    int(response.numel()),
                )
            )
    width = max(int(seq.numel()) for seq in sequences)
    input_ids = torch.full(
        (len(sequences), width), int(pad_token_id), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    for idx, seq in enumerate(sequences):
        length = int(seq.numel())
        input_ids[idx, :length] = seq.to(device=device)
        attention_mask[idx, :length] = 1
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        captured["hidden"] = hidden.detach()
        raise _StopAfterCapture()

    handle = layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            try:
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            except _StopAfterCapture:
                pass
    finally:
        handle.remove()
    hidden = captured.get("hidden")
    if hidden is None:
        raise RuntimeError("failed to capture edit-layer residual activations")
    outputs: list[dict] = [dict(row) for row in rows]
    for seq_idx, (row_idx, arm, question_start, prompt_len, response_len) in enumerate(specs):
        # Train on the actual final-question tokens followed by every fixed
        # response token. Earlier prompt context remains available to the base
        # model but is not itself an activation-reconstruction target.
        stop = prompt_len + response_len
        values = hidden[seq_idx, question_start:stop, :]
        question_tokens = prompt_len - question_start
        expected = question_tokens + response_len
        if int(values.shape[0]) != expected:
            raise RuntimeError("captured activation length does not match question-plus-response length")
        outputs[row_idx][f"{arm}_hidden"] = values.to(dtype=storage_dtype).cpu()
        outputs[row_idx][f"{arm}_question_start"] = int(question_start)
        outputs[row_idx][f"{arm}_question_tokens"] = int(question_tokens)
    for out in outputs:
        if out["control_hidden"].shape != out["intervention_hidden"].shape:
            raise RuntimeError(
                "control and intervention final-question-plus-response activation regions do not align"
            )
        if int(out["control_question_tokens"]) != int(out["intervention_question_tokens"]):
            raise RuntimeError(
                "control and intervention prompts tokenize the final question to different lengths"
            )
        out["question_tokens"] = int(out["control_question_tokens"])
        out["response_tokens"] = int(out["response_ids"].numel())
        out.pop("control_prompt_ids", None)
        out.pop("intervention_prompt_ids", None)
    return outputs


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def _release_failed_capture(device: str) -> None:
    """Release tensors owned by an unwound failed capture before retrying."""
    gc.collect()
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        with torch.cuda.device(torch.device(device)):
            torch.cuda.empty_cache()


def _capture_batch_with_oom_retry(
    model,
    layer,
    rows: list[dict],
    *,
    device: str,
    pad_token_id: int,
    storage_dtype: torch.dtype,
    tokenizer=None,
    capture_fn=None,
) -> list[dict]:
    """Capture one requested batch, recursively halving only on CUDA OOM.

    The caller retains its configured outer batch size. Therefore, once this
    requested batch has completed, the next requested batch starts at the
    original size even if this one required smaller sub-batches.
    """
    if not rows:
        return []
    capture = _capture_batch if capture_fn is None else capture_fn
    retry_parts = None
    try:
        return capture(
            model,
            layer,
            rows,
            tokenizer=tokenizer,
            device=device,
            pad_token_id=pad_token_id,
            storage_dtype=storage_dtype,
        )
    except RuntimeError as exc:
        if not _is_cuda_oom(exc) or len(rows) <= 1:
            raise
        midpoint = max(1, len(rows) // 2)
        retry_parts = (rows[:midpoint], rows[midpoint:])

    if retry_parts is not None:
        left, right = retry_parts
        # Cleanup must happen after leaving the exception handler. Otherwise,
        # the exception traceback can retain tensors from the failed forward
        # while the smaller retry begins.
        _release_failed_capture(device)
        print(
            "[activation-cache] CUDA OOM: retrying requested batch "
            f"of {len(rows)} rows as {len(left)} + {len(right)}",
            flush=True,
        )
        return _capture_batch_with_oom_retry(
            model,
            layer,
            left,
            device=device,
            pad_token_id=pad_token_id,
            storage_dtype=storage_dtype,
            tokenizer=tokenizer,
            capture_fn=capture,
        ) + _capture_batch_with_oom_retry(
            model,
            layer,
            right,
            device=device,
            pad_token_id=pad_token_id,
            storage_dtype=storage_dtype,
            tokenizer=tokenizer,
            capture_fn=capture,
        )
    raise AssertionError("OOM retry planning reached an invalid state")


def _worker(worker_id: int, device: str, model_name: str, layer_index: int, source_root: str, split_row_indices: dict[str, list[int]], out_root: str, batch_size: int, shard_size: int, model_dtype: str, activation_dtype: str, queue) -> None:
    try:
        loaded = load_causal_lm(model_name, device=device, dtype=model_dtype)
        model = loaded.model
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        layer = loaded.layers[int(layer_index)]
        pad_token_id = int(loaded.tokenizer.pad_token_id or 0)
        storage_dtype = torch.bfloat16 if activation_dtype == "bf16" else torch.float16
        manifests: dict[str, list[dict]] = {}
        source_cache_root = Path(source_root)
        for split, row_indices in split_row_indices.items():
            # Load cache rows inside the worker instead of receiving tensor-rich
            # rows from the parent process. This avoids torch multiprocessing
            # creating thousands of shared-memory file descriptors at spawn.
            rows = _load_light_rows_by_index(
                source_cache_root, split, row_indices
            )
            shard_rows: list[dict] = []
            shards: list[dict] = []
            shard_index = 0
            for start in range(0, len(rows), max(1, int(batch_size))):
                batch = rows[start:start + max(1, int(batch_size))]
                built = _capture_batch_with_oom_retry(
                    model, layer, batch, device=device,
                    tokenizer=loaded.tokenizer, pad_token_id=pad_token_id,
                    storage_dtype=storage_dtype,
                )
                shard_rows.extend(built)
                while len(shard_rows) >= int(shard_size):
                    piece, shard_rows = shard_rows[:int(shard_size)], shard_rows[int(shard_size):]
                    name = f"{split}_w{worker_id:02d}_shard_{shard_index:05d}.pt"
                    path = Path(out_root) / name
                    torch.save({"rows": piece}, path)
                    shards.append({
                        "path": name,
                        "num_sequences": len(piece),
                        "sha256": file_sha256(path),
                        "row_index_entries": [
                            {
                                "row_index": int(row["row_index"]),
                                "source_id": str(row["source_id"]),
                                "response_id": str(row["response_id"]),
                                "response_tokens": int(row["response_tokens"]),
                                "question_tokens": int(row["question_tokens"]),
                                "activation_tokens": int(row["control_hidden"].shape[0]),
                            }
                            for row in piece
                        ],
                    })
                    shard_index += 1
                print(
                    f"[activation-cache][{split}][worker {worker_id}] "
                    f"{min(start + len(batch), len(rows))}/{len(rows)}",
                    flush=True,
                )
            if shard_rows:
                name = f"{split}_w{worker_id:02d}_shard_{shard_index:05d}.pt"
                path = Path(out_root) / name
                torch.save({"rows": shard_rows}, path)
                shards.append({
                    "path": name,
                    "num_sequences": len(shard_rows),
                    "sha256": file_sha256(path),
                    "row_index_entries": [
                        {
                            "row_index": int(row["row_index"]),
                            "source_id": str(row["source_id"]),
                            "response_id": str(row["response_id"]),
                            "response_tokens": int(row["response_tokens"]),
                            "question_tokens": int(row["question_tokens"]),
                            "activation_tokens": int(row["control_hidden"].shape[0]),
                        }
                        for row in shard_rows
                    ],
                })
            manifests[split] = shards
        queue.put({"worker": worker_id, "status": "ok", "manifests": manifests, "d_model": resolve_hidden_size(model)})
    except Exception:
        queue.put({"worker": worker_id, "status": "error", "traceback": traceback.format_exc()})


def _expected_metadata(args, source_identity: dict) -> dict:
    # This cache stores raw paired activations, so its validity should not
    # depend on which reconstruction metric is later applied to them.
    return {
        "cache_format": ACTIVATION_CACHE_FORMAT,
        "model_name": str(args.model_name),
        "layer": int(args.layer),
        "model_dtype": str(args.dtype),
        "activation_dtype": str(args.activation_dtype),
        "position_policy": FINAL_QUESTION_RESPONSE_POSITION_POLICY,
        "final_question_prefix": FINAL_QUESTION_PREFIX,
        "source_topk_cache_metadata_sha256": source_identity.get("metadata_sha256"),
    }


def _valid_existing(root: Path, expected: dict) -> bool:
    path = root / "metadata.json"
    if not path.is_file():
        return False
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if any(meta.get(key) != value for key, value in expected.items()):
        return False
    for split in ("train", "val", "test"):
        for shard in (meta.get("splits", {}).get(split, {}).get("shards", [])):
            path = root / str(shard.get("path", ""))
            if not path.is_file():
                return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source_cache_dir", required=True, help="Existing top-k KL cache; fixes the exact row population")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--devices", nargs="+", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--shard_size", type=int, default=64)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--activation_dtype", choices=["bf16", "fp16"], default="bf16")
    args = ap.parse_args()

    source_root = Path(args.source_cache_dir)
    source_identity = cache_identity(source_root, verify_shards=False)
    if source_identity.get("status") != "present":
        raise FileNotFoundError(f"missing source top-k cache metadata: {source_root}")
    expected = _expected_metadata(args, source_identity)
    cache_root = Path(args.cache_dir)
    if cache_root.exists() and _valid_existing(cache_root, expected):
        print(f"[activation-cache] complete cache already exists: {cache_root}")
        return
    if cache_root.exists():
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    devices = _parse_devices(args.devices, args.device)
    # Build only a lightweight assignment plan in the parent. Do not put cached
    # tensors in Process args: under the spawn start method PyTorch shares every
    # tensor storage through IPC, which can exceed the OS open-file limit.
    per_worker: list[dict[str, list[int]]] = [dict() for _ in devices]
    for split in ("train", "val", "test"):
        partitions = _source_index_partitions(source_root, split, len(devices))
        for worker_id, partition in enumerate(partitions):
            per_worker[worker_id][split] = partition

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = []
    for worker_id, device in enumerate(devices):
        process = ctx.Process(
            target=_worker,
            args=(worker_id, device, str(args.model_name), int(args.layer), str(source_root), per_worker[worker_id], str(cache_root), int(args.batch_size), int(args.shard_size), str(args.dtype), str(args.activation_dtype), queue),
        )
        process.start()
        processes.append(process)
    results = [queue.get() for _ in processes]
    for process in processes:
        process.join()
    errors = [row for row in results if row.get("status") != "ok"]
    if errors:
        shutil.rmtree(cache_root, ignore_errors=True)
        raise RuntimeError("activation-cache worker failed:\n" + "\n".join(row.get("traceback", "") for row in errors))
    results.sort(key=lambda row: int(row["worker"]))
    d_models = {int(row["d_model"]) for row in results}
    if len(d_models) != 1:
        raise RuntimeError(f"workers disagreed on d_model: {sorted(d_models)}")
    splits = {}
    for split in ("train", "val", "test"):
        shards = []
        for result in results:
            shards.extend(result["manifests"].get(split, []))
        splits[split] = {
            "num_sequences": int(sum(int(shard["num_sequences"]) for shard in shards)),
            "shards": shards,
        }
    metadata = {
        **expected,
        "d_model": d_models.pop(),
        "source_topk_cache_dir": str(source_root),
        "source_topk_cache_identity": source_identity,
        "requested_splits": ["train", "val", "test"],
        "shard_size": int(args.shard_size),
        "splits": splits,
    }
    (cache_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"activation_cache": "complete", "cache_dir": str(cache_root), "splits": {k: v["num_sequences"] for k, v in splits.items()}}))


if __name__ == "__main__":
    main()
