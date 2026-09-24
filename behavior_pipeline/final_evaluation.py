from __future__ import annotations

"""Shared immutable record contract for final learned-method evaluation."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .artifact_identity import cache_identity
from .topk_kl import aggregate_sequence_kl


TEMPORAL_CACHE_FIELDS = (
    "response_ids",
    "support_ids",
    "support_mask",
    "logp_control",
    "logp_intervention",
    "logp_control_other",
    "logp_intervention_other",
)


def source_clustered_recovery_interval(
    sequence_rows: list[dict],
    *,
    seed: int = 0,
    bootstrap_samples: int = 2000,
) -> dict:
    """Bootstrap recovery by resampling source prompts with all dependent rows."""
    grouped: dict[str, list[dict]] = {}
    for index, row in enumerate(sequence_rows):
        grouped.setdefault(str(row.get("source_id", index)), []).append(row)
    source_ids = sorted(grouped)
    if not source_ids:
        return {
            "bootstrap_unit": "source_id",
            "bootstrap_samples": 0,
            "source_prompts": 0,
            "recovery_bootstrap_95_ci": None,
        }
    rng = np.random.default_rng(int(seed))
    recoveries = []
    for _ in range(max(0, int(bootstrap_samples))):
        sampled = rng.choice(source_ids, size=len(source_ids), replace=True)
        rows = [row for source_id in sampled for row in grouped[str(source_id)]]
        metrics = aggregate_sequence_kl(
            [float(row["source_kl_mean"]) for row in rows],
            [float(row["edited_kl_mean"]) for row in rows],
        )
        recoveries.append(float(metrics["recovery"]))
    return {
        "bootstrap_unit": "source_id",
        "bootstrap_samples": int(len(recoveries)),
        "source_prompts": int(len(source_ids)),
        "recovery_bootstrap_95_ci": (
            [
                float(np.quantile(recoveries, 0.025)),
                float(np.quantile(recoveries, 0.975)),
            ]
            if recoveries
            else None
        ),
    }


def select_prompt_diverse_rows(rows: list[dict], maximum: int) -> list[dict]:
    if int(maximum) <= 0 or len(rows) <= int(maximum):
        return list(rows)
    grouped: dict[str, list[dict]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(str(row.get("source_id", index)), []).append(row)
    selected = []
    depth = 0
    while len(selected) < int(maximum):
        added = False
        for source_id in sorted(grouped):
            candidates = grouped[source_id]
            if depth < len(candidates):
                selected.append(candidates[depth])
                added = True
                if len(selected) >= int(maximum):
                    break
        if not added:
            break
        depth += 1
    return selected


def truncate_cache_row(row: dict, response_token_cap: int) -> dict:
    cap = int(response_token_cap)
    if cap <= 0:
        return dict(row)
    length = min(cap, int(row["response_ids"].shape[0]))
    result = dict(row)
    for key in TEMPORAL_CACHE_FIELDS:
        if key in result:
            result[key] = result[key][:length]
    return result


def prepare_evaluation_rows(
    rows: list[dict], *, maximum: int, response_token_cap: int
) -> list[dict]:
    selected = select_prompt_diverse_rows(rows, int(maximum))
    return [
        truncate_cache_row(row, int(response_token_cap)) for row in selected
    ]


def _manifest_id(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _token_ids_sha256(token_ids) -> str:
    values = [int(value) for value in token_ids.detach().cpu().reshape(-1).tolist()]
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode()
    ).hexdigest()


def build_evaluation_manifest(
    rows: list[dict],
    *,
    cache_dir: str | Path,
    model_name: str,
    layer: int,
    response_token_cap: int,
    maximum: int,
) -> dict:
    identity = cache_identity(cache_dir, verify_shards=True)
    if identity.get("status") != "present":
        raise RuntimeError("final evaluation cache metadata is missing")
    if str(identity.get("model_name")) != str(model_name):
        raise RuntimeError(
            "final evaluation model does not match the teacher cache: "
            f"model={model_name!r} cache={identity.get('model_name')!r}"
        )
    records = []
    for row in rows:
        for direction in ("c_to_i", "i_to_c"):
            records.append(
                {
                    "source_id": str(row["source_id"]),
                    "response_id": str(row["response_id"]),
                    "response_arm": str(row.get("response_arm", "")),
                    "rollout_id": int(row.get("rollout_id", -1)),
                    "direction": direction,
                    "response_tokens": int(row["response_ids"].shape[0]),
                    "response_token_ids_sha256": _token_ids_sha256(
                        row["response_ids"]
                    ),
                }
            )
    base = {
        "metric_unit": "mean conditional KL per response token",
        "evaluation_split": "test",
        "scope": "full_test" if int(maximum) <= 0 else "explicit_subset",
        "max_records": int(maximum),
        "response_token_cap": int(response_token_cap),
        "model_name": str(model_name),
        "layer": int(layer),
        "cache_identity": identity,
        "source_prompts": len({row["source_id"] for row in records}),
        "sequence_direction_records": len(records),
        "records": records,
    }
    return {**base, "manifest_id": _manifest_id(base)}


def validate_evaluation_manifest(manifest: dict) -> None:
    if manifest.get("evaluation_split") != "test":
        raise RuntimeError("final evaluation manifest must use test")
    manifest_id = str(manifest.get("manifest_id", ""))
    base = {key: value for key, value in manifest.items() if key != "manifest_id"}
    if manifest_id != _manifest_id(base):
        raise RuntimeError("evaluation manifest content hash is invalid")


def load_evaluation_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_evaluation_manifest(manifest)
    return manifest
