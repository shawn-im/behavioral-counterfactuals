from __future__ import annotations

"""Activation-reconstruction cache and normalized-MSE utilities."""

import hashlib
import inspect
import json
import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .numerics import reject_invalid


ACTIVATION_CACHE_FORMAT = "counterfactual_activation_reconstruction"

ACTIVATION_NMSE_DEFINITION = "sequence_mean_of_token_feature_mse_over_l2_target_activation_norm"


def _metadata_sha256(cache_dir: Path) -> str:
    return hashlib.sha256((cache_dir / "metadata.json").read_bytes()).hexdigest()


def _load_shard(path: Path) -> dict:
    kwargs = {"map_location": "cpu"}
    if "mmap" in inspect.signature(torch.load).parameters:
        kwargs["mmap"] = True
    return torch.load(path, **kwargs)


def _descriptor(shard_path: str, local_index: int, row: dict) -> dict:
    return {
        "shard_path": str(shard_path),
        "local_index": int(local_index),
        "row_index": int(row.get("row_index", local_index)),
        "source_id": str(row["source_id"]),
        "response_id": str(row["response_id"]),
        "response_tokens": int(row["response_tokens"]),
        "question_tokens": int(row["question_tokens"]),
        "activation_tokens": int(row["control_hidden"].shape[0]),
    }


def ensure_activation_cache_indices(
    cache_dir: str | Path,
    *,
    splits: tuple[str, ...] = ("train", "val", "test"),
    progress: bool = False,
) -> dict[str, Path]:
    """Build or reuse split row indices mapping examples to shard locations.

    Prefer descriptors embedded in metadata. Otherwise load one shard at a time
    and extract descriptors, without retaining the full cache's activations.
    Validate row counts and uniqueness when building an index, and reuse an
    index when its metadata fingerprint, split, and expected row count match.
    """
    root = Path(cache_dir)
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("cache_format") != ACTIVATION_CACHE_FORMAT:
        raise ValueError(
            f"activation cache has format {metadata.get('cache_format')!r}; "
            f"expected {ACTIVATION_CACHE_FORMAT!r}"
        )
    metadata_hash = _metadata_sha256(root)
    outputs = {}
    for split in splits:
        split_meta = (metadata.get("splits") or {}).get(split)
        if not isinstance(split_meta, dict):
            raise ValueError(f"activation cache has no split {split!r}")
        output = root / f"{split}_row_index.json"
        declared_count = split_meta.get("num_sequences")
        if declared_count is not None:
            expected_count: int | None = int(declared_count)
        elif all("num_sequences" in shard for shard in split_meta.get("shards", [])):
            expected_count = sum(
                int(shard.get("num_sequences", 0))
                for shard in split_meta.get("shards", [])
            )
        else:
            expected_count = None
        if output.is_file():
            try:
                existing = json.loads(output.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
            if (
                existing.get("metadata_sha256") == metadata_hash
                and str(existing.get("split")) == split
                and (
                    expected_count is None
                    or len(existing.get("rows", [])) == expected_count
                )
            ):
                outputs[split] = output
                continue

        shards = list(split_meta.get("shards", []))
        descriptors = []
        report_every = max(1, len(shards) // 20)
        for shard_number, shard in enumerate(shards, start=1):
            shard_path = str(shard["path"])
            embedded = shard.get("row_index_entries")
            if isinstance(embedded, list) and len(embedded) == int(
                shard.get("num_sequences", len(embedded))
            ):
                descriptors.extend(
                    {
                        **entry,
                        "shard_path": shard_path,
                        "local_index": int(local_index),
                    }
                    for local_index, entry in enumerate(embedded)
                )
            else:
                payload = _load_shard(root / shard_path)
                rows = payload.get("rows", [])
                descriptors.extend(
                    _descriptor(shard_path, local_index, row)
                    for local_index, row in enumerate(rows)
                )
                del rows, payload
            if progress and (
                shard_number == 1
                or shard_number == len(shards)
                or shard_number % report_every == 0
            ):
                print(
                    f"[activation-index][{split}] {shard_number}/{len(shards)} shards",
                    flush=True,
                )
        descriptors.sort(key=lambda row: int(row["row_index"]))
        if expected_count is not None and len(descriptors) != expected_count:
            raise RuntimeError(
                f"activation index for {split} has {len(descriptors)} rows; "
                f"metadata declares {expected_count}"
            )
        row_indices = [int(row["row_index"]) for row in descriptors]
        if len(row_indices) != len(set(row_indices)):
            raise RuntimeError(f"activation index for {split} has duplicate row indices")
        payload = {
            "metadata_sha256": metadata_hash,
            "split": split,
            "rows": descriptors,
        }
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(output)
        outputs[split] = output
    return outputs


class ActivationReconstructionDataset(Dataset):
    """Paired activations with one shared shard cache per process.

    Dataset instances share the most recently loaded shard. Returned rows hold
    tensor references, so active batches may keep tensors from other shards alive.
    """

    _shared_loaded_path: Path | None = None
    _shared_loaded_rows: list[dict] | None = None

    def __init__(self, cache_dir: str | Path, split: str):
        self.cache_dir = Path(cache_dir)
        self.split = str(split)
        self.meta = json.loads(
            (self.cache_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if self.meta.get("cache_format") != ACTIVATION_CACHE_FORMAT:
            raise ValueError(
                f"activation cache has format {self.meta.get('cache_format')!r}; "
                f"expected {ACTIVATION_CACHE_FORMAT!r}"
            )
        split_names = set((self.meta.get("splits") or {}).keys())
        if split_names != {"train", "val", "test"}:
            raise ValueError(
                "activation-reconstruction caches must contain exactly train, val, and test; "
                f"found {sorted(split_names)}"
            )
        if self.split not in split_names:
            raise ValueError(f"activation cache has no split {self.split!r}")
        index_path = ensure_activation_cache_indices(
            self.cache_dir, splits=(self.split,), progress=False
        )[self.split]
        self.index = json.loads(index_path.read_text(encoding="utf-8"))["rows"]
        self.source_splits = (self.split,)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        descriptor = self.index[int(idx)]
        shard_path = (self.cache_dir / str(descriptor["shard_path"])).resolve()
        cls = type(self)
        if cls._shared_loaded_path != shard_path or cls._shared_loaded_rows is None:
            payload = _load_shard(shard_path)
            cls._shared_loaded_rows = payload.get("rows", [])
            cls._shared_loaded_path = shard_path
        row = dict(cls._shared_loaded_rows[int(descriptor["local_index"])])
        row.setdefault("cache_source_split", self.split)
        return row

    def source_subset_indices(
        self, *, fraction: float, seed: int
    ) -> tuple[list[int], list[str]]:
        fraction = float(fraction)
        if fraction <= 0.0 or not self.index:
            return [], []
        source_ids = sorted({str(row["source_id"]) for row in self.index})
        rng = random.Random(int(seed) + 104729)
        rng.shuffle(source_ids)
        count = max(1, int(round(len(source_ids) * min(1.0, fraction))))
        selected_ids = sorted(source_ids[:count])
        selected = set(selected_ids)
        return (
            [
                index
                for index, row in enumerate(self.index)
                if str(row["source_id"]) in selected
            ],
            selected_ids,
        )

    def shard_shuffled_indices(self, *, seed: int) -> list[int]:
        """Shuffle shards and rows within shards while keeping reads sequential."""
        grouped: dict[str, list[int]] = {}
        for index, row in enumerate(self.index):
            grouped.setdefault(str(row["shard_path"]), []).append(index)
        rng = random.Random(int(seed))
        shard_paths = list(grouped)
        rng.shuffle(shard_paths)
        result = []
        for shard_path in shard_paths:
            indices = grouped[shard_path]
            rng.shuffle(indices)
            result.extend(indices)
        return result


def token_nmse(prediction: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Return one normalized MSE per token, averaged over the feature dimension.

    Compute mean((prediction-target)^2) / max(||target||_2, eps) in float32.
    The denominator is the target L2 norm, not its square. Reject nonfinite
    inputs or results; averaging across tokens and examples is done by callers.
    """

    prediction = prediction.float()
    target = target.float()
    reject_invalid(prediction, ~torch.isfinite(prediction), "activation NMSE prediction")
    reject_invalid(target, ~torch.isfinite(target), "activation NMSE target")
    numerator = (prediction - target).square().mean(dim=-1)
    denominator = target.square().sum(dim=-1).sqrt().clamp_min(float(eps))
    reject_invalid(numerator, ~torch.isfinite(numerator), "activation NMSE squared error")
    reject_invalid(denominator, ~torch.isfinite(denominator), "activation NMSE target norm")
    result = numerator / denominator
    reject_invalid(result, ~torch.isfinite(result), "activation NMSE")
    return result


def activation_recovery(source_nmse: float, edited_nmse: float) -> float:
    if source_nmse <= 0.0:
        return 0.0
    return 1.0 - float(edited_nmse) / float(source_nmse)


def row_identity(row: dict) -> tuple[str, str]:
    return str(row["source_id"]), str(row["response_id"])
