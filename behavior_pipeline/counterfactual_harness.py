from __future__ import annotations

"""Configuration, manifests, hashing, and resumability for counterfactual analysis."""

import hashlib
import json
import os
import random
import shutil
import copy
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
import torch

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised in minimal environments
    raise ImportError(
        "scripts/run_counterfactual_analysis.py requires PyYAML; install requirements.txt"
    ) from exc


def prune_generated_directory(directory: Path, expected_names: set[str]) -> None:
    """Remove generated files not listed among the expected output names."""
    if not directory.exists():
        return
    for child in directory.iterdir():
        if child.name in expected_names:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


DEFAULT_CONFIG = {
    "run": {
        "name": "counterfactual_analysis",
        "output_dir": "runs/analysis/counterfactual_analysis",
        "seed": 0,
        "resume": True,
    },
    "model": {
        "name": "",
        "layer": 20,
        "dtype": "bf16",
        "device": "cuda:0",
        "devices": [],
    },
    "edit": {
        "checkpoint": "",
        "directions": "both",
    },
    "data": {
        "cache_dir": "",
        "scruples_dir": "",
        "evaluation_split": "test",
        "max_examples": 0,
        "analysis_response_token_cap": 0,
        "static_runs_dir": "",
    },
    "teacher_forced": {
        "enabled": True,
        "min_source_kl": 0.0,
        "batch_size": 256,
    },
    "attribution": {
        "enabled": True,
        "methods": ["ig"],
        "primary_method": "ig",
        "ig_steps": 16,
        "primary_interpretation_mode": "teacher_forced",
        "modes": ["teacher_forced"],
        "integration_batch_size": 16,
        "support_only_projection": True,
        "support_projection_chunk_size": 2048,
        "max_records": 0,
        "qualitative_fraction": 0.01,
    },
    "ablations": {
        "edit_events": True,
        "batch_size": 16,
        "max_records": 0,
        "top_events_per_method": 5,
        "random_events_per_method": 5,
        "support_only_projection": True,
        "support_projection_chunk_size": 2048,
    },
    "phrases": {
        "max_span_tokens": 5,
        "top_positive": 20,
        "top_negative": 20,
        "deduplicate_overlap": True,
    },
    "native_analysis": {
        "enabled": True,
        "directions": ["c_to_i", "i_to_c"],
        "one_record_per_source_direction": True,
        "max_records": 0,
        "patch_sites": [
            "filler",
            "question_span",
            "question_and_filler",
        ],
        "top_tokens_per_position": 8,
        "top_positions_per_record": 20,
        "report_examples": 20,
        "bootstrap_samples": 1000,
        "native_record_batch_size": 2,
        "native_condition_batch_size": 16,
    },
    "exhaustive_validation": {
        "enabled": True,
        "records": 10,
        "ablation_batch_size": 16,
    },
    "sentence_browser": {
        "enabled": True,
        "records": 500,
        "keywords": ["user", "suggest", "suggested", "suggests"],
    },
    "report": {
        "generate_html": True,
        "attribution_view": "comparison",
        "primary_attribution_method": "ig",
        "top_token_fraction": 0.05,
        "top_token_limit": 30,
        "qualitative_probability_token_limit": 20,
        "objective_comparison_path": "",
    },
}


STAGE_NAMES = (
    "manifest",
    "teacher_forced",
    "downstream_attribution",
    "causal_ablations",
    "native_hypothesis_analysis",
    "exhaustive_token_validation",
    "keyword_sentence_browser",
    "report",
)


def _deep_merge(base: dict, override: dict) -> dict:
    out = {}
    for key in base.keys() | override.keys():
        left = base.get(key)
        right = override.get(key)
        if isinstance(left, dict) and isinstance(right, dict):
            out[key] = _deep_merge(left, right)
        elif key in override:
            out[key] = right
        else:
            out[key] = left
    return out


def load_config(
    path: str | Path,
    *,
    overrides: dict | None = None,
) -> dict:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    _validate_known_keys(DEFAULT_CONFIG, raw)
    config = _deep_merge(DEFAULT_CONFIG, raw)
    if overrides:
        overrides = copy.deepcopy(overrides)
        _validate_known_keys(DEFAULT_CONFIG, overrides)
        config = _deep_merge(config, overrides)
    config["_config_path"] = str(path.resolve())
    validate_config(config)
    return config


def topological_stage_order(
    selected: Iterable[int],
    dependencies: Mapping[int, Iterable[int]],
) -> list[int]:
    """Return selected stages ordered so every selected dependency runs first."""

    selected_set = {int(index) for index in selected}
    ordered: list[int] = []
    remaining = set(selected_set)
    while remaining:
        ready = sorted(
            index
            for index in remaining
            if all(
                int(dependency) not in selected_set or int(dependency) in ordered
                for dependency in dependencies.get(index, ())
            )
        )
        if not ready:
            unresolved = {
                index: tuple(
                    int(dependency)
                    for dependency in dependencies.get(index, ())
                    if int(dependency) in remaining
                )
                for index in sorted(remaining)
            }
            raise RuntimeError(f"Cyclic stage dependencies: {unresolved}")
        # Prefer the lowest-numbered ready stage, then recompute readiness.
        # This preserves intuitive numeric execution order whenever dependencies
        # allow it instead of eagerly scheduling every currently-ready stage.
        index = ready[0]
        ordered.append(index)
        remaining.remove(index)
    return ordered


def override_model_devices(config: dict, values: list[str] | None) -> dict:
    """Apply a CLI GPU override to the resolved harness configuration."""
    if not values:
        return config
    devices = [
        device.strip()
        for value in values
        for device in str(value).split(",")
        if device.strip()
    ]
    if not devices:
        raise ValueError("Device override must contain at least one device")
    config["model"]["device"] = devices[0]
    config["model"]["devices"] = devices if len(devices) > 1 else []
    validate_config(config)
    return config


def _validate_known_keys(reference: dict, value: dict, prefix: str = "") -> None:
    unknown = sorted(set(value) - set(reference))
    if unknown:
        names = [f"{prefix}{key}" for key in unknown]
        raise ValueError("Unknown configuration values: " + ", ".join(names))
    for key, child in value.items():
        expected = reference.get(key)
        if isinstance(child, dict) and isinstance(expected, dict):
            _validate_known_keys(expected, child, f"{prefix}{key}.")


def validate_config(config: dict) -> None:
    required = {
        "model.name": config["model"].get("name"),
        "edit.checkpoint": config["edit"].get("checkpoint"),
        "data.cache_dir": config["data"].get("cache_dir"),
        "run.output_dir": config["run"].get("output_dir"),
    }
    missing = [key for key, value in required.items() if not str(value or "").strip()]
    if missing:
        raise ValueError("Missing required configuration values: " + ", ".join(missing))
    if config["edit"]["directions"] not in {"both", "c_to_i", "i_to_c"}:
        raise ValueError("edit.directions must be both, c_to_i, or i_to_c")
    methods = config["attribution"].get("methods", [])
    if not isinstance(methods, list) or not methods:
        raise ValueError("attribution.methods must be a non-empty list")
    if len(set(str(method) for method in methods)) != len(methods):
        raise ValueError("attribution.methods must not contain duplicates")
    valid_methods = {"ig", "relp"}
    if any(str(method) not in valid_methods for method in methods):
        raise ValueError("attribution.methods entries must be ig or relp")
    if str(config["attribution"].get("primary_method")) not in {str(method) for method in methods}:
        raise ValueError("attribution.primary_method must be included in attribution.methods")
    report_view = str(config.get("report", {}).get("attribution_view", "comparison"))
    if report_view not in {"comparison", "primary"}:
        raise ValueError("report.attribution_view must be comparison or primary")
    report_primary = str(
        config.get("report", {}).get(
            "primary_attribution_method",
            config["attribution"].get("primary_method", methods[0]),
        )
    )
    if report_primary not in {str(method) for method in methods}:
        raise ValueError("report.primary_attribution_method must be included in attribution.methods")
    native = config.get("native_analysis", {})
    native_directions = [str(value) for value in native.get("directions", [])]
    if not native_directions or any(value not in {"c_to_i", "i_to_c"} for value in native_directions):
        raise ValueError("native_analysis.directions entries must be c_to_i or i_to_c")
    valid_patch_sites = {
        "filler",
        "question_span",
        "question_and_filler",
        "newlines_and_punctuation",
        "question_and_newlines_and_punctuation",
    }
    patch_sites = [str(value) for value in native.get("patch_sites", [])]
    if not patch_sites or any(value not in valid_patch_sites for value in patch_sites):
        raise ValueError("native_analysis.patch_sites contains an unknown patch site")
    for key in (
        "top_tokens_per_position",
        "bootstrap_samples",
        "native_record_batch_size",
        "native_condition_batch_size",
    ):
        if int(native.get(key, 0)) <= 0:
            raise ValueError(f"native_analysis.{key} must be positive")
    for path_key in ("edit.checkpoint", "data.cache_dir"):
        section, key = path_key.split(".")
        path = Path(str(config[section][key]))
        if not path.exists():
            raise FileNotFoundError(f"{path_key} does not exist: {path}")
    if config["data"].get("evaluation_split") != "test":
        raise ValueError("data.evaluation_split must be test")
    qualitative_fraction = float(config["attribution"].get("qualitative_fraction", 0.01))
    if not (0.0 < qualitative_fraction <= 1.0):
        raise ValueError("attribution.qualitative_fraction must be in (0, 1]")
    top_token_fraction = float(config.get("report", {}).get("top_token_fraction", 0.05))
    if not (0.0 < top_token_fraction <= 1.0):
        raise ValueError("report.top_token_fraction must be in (0, 1]")
    for section, key in (
        ("data", "max_examples"),
        ("data", "analysis_response_token_cap"),
        ("ablations", "max_records"),
        ("attribution", "max_records"),
        ("report", "top_token_limit"),
        ("native_analysis", "max_records"),
        ("native_analysis", "top_positions_per_record"),
        ("native_analysis", "report_examples"),
    ):
        if int(config[section][key]) < 0:
            raise ValueError(f"{section}.{key} must be nonnegative")
    for section, key in (
        ("report", "qualitative_probability_token_limit"),
        ("teacher_forced", "batch_size"),
        ("ablations", "batch_size"),
        ("attribution", "integration_batch_size"),
        ("attribution", "ig_steps"),
        ("ablations", "top_events_per_method"),
        ("ablations", "random_events_per_method"),
        ("ablations", "support_projection_chunk_size"),
        ("exhaustive_validation", "ablation_batch_size"),
        ("exhaustive_validation", "records"),
        ("sentence_browser", "records"),
    ):
        if int(config[section][key]) <= 0:
            raise ValueError(f"{section}.{key} must be positive")
    browser_keywords = config.get("sentence_browser", {}).get("keywords", [])
    if not isinstance(browser_keywords, list) or not browser_keywords or any(
        not str(value).strip() for value in browser_keywords
    ):
        raise ValueError("sentence_browser.keywords must be a non-empty list")
    if bool(config.get("exhaustive_validation", {}).get("enabled", True)) and not bool(
        config.get("attribution", {}).get("enabled", True)
    ):
        raise ValueError("exhaustive_validation requires attribution.enabled=true")
    if int(config["attribution"]["support_projection_chunk_size"]) <= 0:
        raise ValueError("attribution.support_projection_chunk_size must be positive")
    if config["attribution"]["primary_interpretation_mode"] != "teacher_forced":
        raise ValueError("invalid attribution.primary_interpretation_mode")
    attribution_modes = config["attribution"].get("modes", [])
    if attribution_modes != ["teacher_forced"]:
        raise ValueError("attribution.modes must be ['teacher_forced']")
    devices = config["model"].get("devices", [])
    if not isinstance(devices, list):
        raise ValueError("model.devices must be a list")
    if any(not str(device).strip() for device in devices):
        raise ValueError("model.devices entries must be non-empty")


def directions(config: dict) -> list[str]:
    value = str(config["edit"]["directions"])
    return ["c_to_i", "i_to_c"] if value == "both" else [value]


def set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _path_record(path: Path, *, content_hash: bool) -> dict:
    stat = path.stat()
    row = {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if content_hash and path.is_file():
        row["sha256"] = sha256_file(path)
    return row


def path_fingerprint(path: str | Path, *, content_hash: bool = False) -> dict:
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "missing": True}
    if path.is_file():
        return _path_record(path, content_hash=content_hash)
    rows = []
    for child in sorted(path.rglob("*")):
        if not child.is_file():
            continue
        relative = child.relative_to(path)
        stat = child.stat()
        row = {
            "path": str(relative),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
        if content_hash:
            row["sha256"] = sha256_file(child)
        rows.append(row)
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "path": str(path.resolve()),
        "files": len(rows),
        "digest": hashlib.sha256(payload).hexdigest(),
    }


def hash_object(value) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def output_dir(config: dict) -> Path:
    return Path(str(config["run"]["output_dir"]))


def stage_dir(config: dict, stage_index: int) -> Path:
    name = STAGE_NAMES[int(stage_index)]
    return output_dir(config) / "stages" / f"stage_{int(stage_index):02d}_{name}"


def write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _output_records(paths: Iterable[Path]) -> list[dict]:
    records = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_file():
            records.append(_path_record(path, content_hash=True))
        else:
            records.append(path_fingerprint(path, content_hash=False))
    return records


def _scientific_config_snapshot(config: dict, sections: Iterable[str]) -> dict:
    """Copy the stage-relevant configuration, excluding GPU placement.

    Device assignments determine where work runs and do not invalidate cached
    results. Other settings in the selected sections remain fingerprinted.
    """

    selected = {section: copy.deepcopy(config.get(section)) for section in sections}
    model = selected.get("model")
    if isinstance(model, dict):
        model.pop("device", None)
        model.pop("devices", None)
    return selected


class StageRunner:
    """Execute numbered stages with input-hash validation and durable statuses."""

    def __init__(self, config: dict, *, root: Path, force: bool = False):
        self.config = config
        self.root = Path(root)
        self.force = bool(force)
        self.run_dir = output_dir(config)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def status_path(self, index: int) -> Path:
        return stage_dir(self.config, index) / "stage_status.json"

    def last_complete_status_path(self, index: int) -> Path:
        return stage_dir(self.config, index) / "stage_status.last_complete.json"

    def dependency_digest(self, dependencies: Iterable[int]) -> list[dict]:
        out = []
        for index in dependencies:
            status = read_json(self.status_path(index), {})
            out.append(
                {
                    "stage": int(index),
                    "name": STAGE_NAMES[int(index)],
                    "status": status.get("status"),
                    "output_hash": status.get("output_hash"),
                }
            )
        return out

    def run(
        self,
        index: int,
        function: Callable[[], Iterable[Path] | dict | None],
        *,
        dependencies: Iterable[int] = (),
        config_sections: Iterable[str] = (),
        external_inputs: Iterable[Path] = (),
        force: bool | None = None,
    ) -> bool:
        index = int(index)
        sdir = stage_dir(self.config, index)
        sdir.mkdir(parents=True, exist_ok=True)
        dependency_rows = self.dependency_digest(dependencies)
        failed_dependencies = [
            row for row in dependency_rows if str(row.get("status")) != "complete"
        ]
        if failed_dependencies:
            raise RuntimeError(
                f"Stage {index} dependencies are incomplete: {failed_dependencies}"
            )
        selected_config = _scientific_config_snapshot(self.config, config_sections)
        input_payload = {
            "config": selected_config,
            "dependencies": dependency_rows,
            "external_inputs": [
                path_fingerprint(path, content_hash=path.is_file())
                for path in external_inputs
            ],
        }
        input_hash = hash_object(input_payload)
        previous = read_json(self.status_path(index), {})
        last_complete = read_json(self.last_complete_status_path(index), {})
        should_force = self.force if force is None else bool(force)

        def reusable_status(status: dict) -> tuple[bool, bool]:
            outputs_exist = all(
                Path(row["path"]).exists() for row in status.get("outputs", [])
            )
            exact = status.get("input_hash") == input_hash
            compatible = status.get("inputs", {}) == input_payload
            reusable = (
                status.get("status") == "complete"
                and outputs_exist
                and (exact or compatible)
            )
            return reusable, exact

        resume_enabled = bool(self.config["run"].get("resume", True)) and not should_force
        resume_source = None
        exact_resume = False
        if resume_enabled:
            for candidate in (previous, last_complete):
                reusable, exact = reusable_status(candidate)
                if reusable:
                    resume_source = candidate
                    exact_resume = exact
                    break

        if resume_source is not None:
            restored = copy.deepcopy(resume_source)
            if not exact_resume:
                restored["input_hash"] = input_hash
                restored["inputs"] = input_payload
                restored["resume_fingerprint_migrated_at_unix"] = time.time()
            write_json(self.status_path(index), restored)
            write_json(self.last_complete_status_path(index), restored)
            print(f"[harness] stage {index} {STAGE_NAMES[index]}: reuse", flush=True)
            return False

        # Preserve a known-good completion before a scientifically different
        # attempt replaces the live status with "running".
        if previous.get("status") == "complete":
            write_json(self.last_complete_status_path(index), previous)

        started = time.time()
        running = {
            "stage": index,
            "name": STAGE_NAMES[index],
            "status": "running",
            "input_hash": input_hash,
            "inputs": input_payload,
            "started_at_unix": started,
            "pid": os.getpid(),
            "warnings": [],
            "failures": [],
        }
        write_json(self.status_path(index), running)
        print(f"[harness] stage {index} {STAGE_NAMES[index]}: start", flush=True)
        try:
            result = function() or []
            if isinstance(result, dict):
                outputs = [Path(path) for path in result.get("outputs", [])]
                warnings = [str(value) for value in result.get("warnings", [])]
            else:
                outputs = [Path(path) for path in result]
                warnings = []
            missing = [str(path) for path in outputs if not path.exists()]
            if missing:
                raise RuntimeError(f"Stage declared missing outputs: {missing}")
            output_records = _output_records(outputs)
            completed = {
                **running,
                "status": "complete",
                "completed_at_unix": time.time(),
                "elapsed_seconds": time.time() - started,
                "outputs": output_records,
                "output_hash": hash_object(output_records),
                "warnings": warnings,
            }
            write_json(self.status_path(index), completed)
            write_json(self.last_complete_status_path(index), completed)
            print(
                f"[harness] stage {index} {STAGE_NAMES[index]}: complete "
                f"({completed['elapsed_seconds']:.1f}s)",
                flush=True,
            )
            return True
        except BaseException as exc:
            failed = {
                **running,
                "status": "failed",
                "completed_at_unix": time.time(),
                "elapsed_seconds": time.time() - started,
                "failures": [{"type": type(exc).__name__, "message": str(exc)}],
            }
            write_json(self.status_path(index), failed)
            raise


@contextmanager
def stage_seed(config: dict, stage_index: int):
    """Set a deterministic seed for a block; do not restore random states on exit.

    The index is a fixed seed namespace supplied by the caller, independent of
    the public stage numbering.
    """
    seed = int(config["run"]["seed"]) + int(stage_index) * 100003
    set_all_seeds(seed)
    yield seed
