from __future__ import annotations

"""Content identities used by resumable training and final evaluation."""

import hashlib
import json
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_identity(
    cache_dir: str | Path, *, verify_shards: bool = False
) -> dict:
    root = Path(cache_dir)
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        return {
            "cache_dir": str(root.resolve()),
            "status": "missing_metadata",
        }
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    shards = []
    for split, split_metadata in sorted(metadata.get("splits", {}).items()):
        for shard in split_metadata.get("shards", []):
            relative_path = str(shard.get("path", ""))
            declared_sha256 = shard.get("sha256")
            actual_sha256 = None
            if verify_shards:
                shard_path = root / relative_path
                if not shard_path.is_file():
                    raise RuntimeError(
                        f"cache shard declared in metadata is missing: {shard_path}"
                    )
                actual_sha256 = file_sha256(shard_path)
                if declared_sha256 != actual_sha256:
                    raise RuntimeError(
                        "cache shard hash does not match metadata: "
                        f"{shard_path}"
                    )
            shards.append(
                {
                    "split": str(split),
                    "path": relative_path,
                    "sha256": (
                        actual_sha256 if verify_shards else declared_sha256
                    ),
                    "num_sequences": shard.get("num_sequences"),
                }
            )
    return {
        "cache_dir": str(root.resolve()),
        "status": "present",
        "model_name": metadata.get("model_name"),
        "top_k": metadata.get("top_k"),
        "generation_metadata_sha256": metadata.get(
            "generation_metadata_sha256"
        ),
        "metadata_sha256": file_sha256(metadata_path),
        "shards": shards,
    }


def checkpoint_identity(path: str | Path) -> dict:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        return {
            "path": str(checkpoint.resolve()),
            "status": "missing",
        }
    return {
        "path": str(checkpoint.resolve()),
        "status": "present",
        "sha256": file_sha256(checkpoint),
    }
