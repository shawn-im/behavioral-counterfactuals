from __future__ import annotations

"""Configuration and execution for counterfactual edit-module training and sanity checks."""

import hashlib
import json
import math
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from tqdm.auto import tqdm

from .artifact_identity import cache_identity

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The training harness requires PyYAML; install requirements.txt"
    ) from exc


def fixed_source_eval_rows(
    rows: list[dict], *, fraction: float, seed: int
) -> tuple[list[dict], list[str]]:
    """Select a deterministic source-complete subset for train diagnostics."""
    import random

    fraction = float(fraction)
    if fraction <= 0.0 or not rows:
        return [], []
    source_ids = sorted({str(row.get("source_id", "")) for row in rows})
    if not source_ids:
        return [], []
    rng = random.Random(int(seed) + 104729)
    rng.shuffle(source_ids)
    count = max(1, int(round(len(source_ids) * min(1.0, fraction))))
    selected = set(source_ids[:count])
    return (
        [row for row in rows if str(row.get("source_id", "")) in selected],
        sorted(selected),
    )


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _structured_child_event(line: str) -> tuple[str, dict] | None:
    """Parse the trainer's one-key JSON event envelope."""
    text = str(line).strip()
    if not text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or len(value) != 1:
        return None
    event, payload = next(iter(value.items()))
    if not isinstance(payload, dict):
        return None
    return str(event), payload


def _format_live_child_line(line: str) -> str | None:
    """Format structured trainer events for concise live console output."""
    text = str(line).strip()
    if not text:
        return None
    parsed = _structured_child_event(text)
    if parsed is None:
        return text
    event, payload = parsed

    def number(key: str, *, scientific: bool = False) -> str | None:
        raw = payload.get(key)
        if raw is None:
            return None
        try:
            x = float(raw)
        except (TypeError, ValueError):
            return str(raw)
        return f"{x:.3e}" if scientific else f"{x:.6f}"

    if event == "topk_kl_distill_eval":
        parts = [
            f"eval split={payload.get('split', '?')}",
            f"epoch={payload.get('epoch', '?')}",
            f"update={payload.get('update_step', '?')}",
        ]
        for label, key in (("loss", "loss"), ("recovery", "recovery")):
            formatted = number(key)
            if formatted is not None:
                parts.append(f"{label}={formatted}")
        return " ".join(parts)

    if event == "topk_kl_distill_train_update":
        parts = [
            "train",
            f"epoch={payload.get('epoch', '?')}",
            f"update={payload.get('update_step', '?')}",
        ]
        for label, key, sci in (
            ("loss", "loss", False),
            ("recovery", "recovery", False),
            ("lr", "learning_rate", True),
            ("grad", "gradient_global_norm", True),
        ):
            formatted = number(key, scientific=sci)
            if formatted is not None:
                parts.append(f"{label}={formatted}")

        return " ".join(parts)

    if event == "topk_kl_distill_train_microbatch":
        parts = [
            "train",
            f"epoch={payload.get('epoch', '?')}",
            f"microbatch={payload.get('batch', '?')}/{payload.get('batches', '?')}",
            f"update={payload.get('update_step', '?')}",
            (
                "accum="
                f"{payload.get('microbatches_in_update', '?')}/"
                f"{payload.get('microbatches_per_update', '?')}"
            ),
        ]
        formatted = number("loss")
        if formatted is not None:
            parts.append(f"loss={formatted}")
        formatted = number("recovery")
        if formatted is not None:
            parts.append(f"recovery={formatted}")
        return " ".join(parts)

    return text


def load_training_harness_config(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("training harness config must be a YAML mapping")
    for section in ("run", "training", "sweep"):
        if not isinstance(value.get(section), dict):
            raise ValueError(f"missing mapping section: {section}")
    if not isinstance(value.get("variants"), list):
        raise ValueError("missing list section: variants")

    run = value["run"]
    training = value["training"]
    sweep = value["sweep"]
    if not str(run.get("output_root", "")).strip():
        raise ValueError("run.output_root must be set")
    for key in ("cache_dir", "model_name"):
        if not str(training.get(key, "")).strip():
            raise ValueError(f"training.{key} must be set")

    allowed_training_keys = {
        "batch_size",
        "cache_check_max_rows",
        "cache_dir",
        "directions",
        "dtype",
        "epochs",
        "eval_batch_size",
        "forward_batch_size",
        "grad_accum_steps",
        "input_norm_mode",
        "layer",
        "log_every_microbatches",
        "log_every_updates",
        "model_name",
        "seed",
        "support_projection_chunk_size",
        "train_eval_fraction",
        "training_support_only_projection",
    }
    unknown_training_keys = sorted(set(training) - allowed_training_keys)
    if unknown_training_keys:
        raise ValueError(
            "unknown production training keys: " + ", ".join(unknown_training_keys)
        )

    input_norm_mode = str(training.get("input_norm_mode", "rms"))
    if input_norm_mode != "rms":
        raise ValueError(f"unsupported training.input_norm_mode: {input_norm_mode!r}")
    if int(training.get("epochs", 2)) <= 0:
        raise ValueError("training.epochs must be positive")
    for key in ("batch_size", "eval_batch_size", "forward_batch_size"):
        if int(training.get(key, 1)) <= 0:
            raise ValueError(f"training.{key} must be positive")
    if int(training.get("grad_accum_steps", 1)) <= 0:
        raise ValueError("training.grad_accum_steps must be positive")
    if int(training.get("support_projection_chunk_size", 2048)) <= 0:
        raise ValueError(
            "training.support_projection_chunk_size must be positive"
        )
    if int(training.get("log_every_microbatches", 0)) < 0:
        raise ValueError(
            "training.log_every_microbatches must be nonnegative"
        )
    train_eval_fraction = float(training.get("train_eval_fraction", 0.10))
    if not 0.0 <= train_eval_fraction <= 1.0:
        raise ValueError("training.train_eval_fraction must be in [0, 1]")

    learning_rates = sweep.get("learning_rates")
    if not isinstance(learning_rates, list) or not learning_rates:
        raise ValueError("sweep.learning_rates must be a non-empty list")
    parsed_lrs = [float(value) for value in learning_rates]
    if any(lr <= 0.0 or not math.isfinite(lr) for lr in parsed_lrs):
        raise ValueError("sweep.learning_rates entries must be finite and positive")
    if len(set(parsed_lrs)) != len(parsed_lrs):
        raise ValueError("sweep.learning_rates must not contain duplicates")
    if str(sweep.get("selection_metric", "")) != "val_recovery":
        raise ValueError("sweep.selection_metric must be val_recovery")

    methods = training_methods(value)
    expected = {"fixed_steering", "directional_linear", "base_mlp"}
    kinds = {row["edit_kind"] for row in methods}
    if len(methods) != 3 or kinds != expected:
        raise ValueError(
            "variants must contain exactly fixed_steering, directional_linear, and base_mlp"
        )
    primary = [row for row in methods if bool(row.get("primary", False))]
    if len(primary) != 1 or primary[0]["edit_kind"] != "base_mlp":
        raise ValueError("base_mlp must be the single primary editor")

    # Expand once during validation so malformed sweep rows fail before any
    # workers are launched.
    expanded = expand_training_variants(value)
    if len(expanded) != len(methods) * len(parsed_lrs):
        raise ValueError("learning-rate sweep expansion is inconsistent")

    final_evaluation = value.get("final_evaluation")
    if not isinstance(final_evaluation, dict):
        raise ValueError("missing mapping section: final_evaluation")
    if final_evaluation.get("evaluation_split") != "test":
        raise ValueError("final_evaluation.evaluation_split must be test")
    for key in ("max_records", "response_token_cap"):
        if int(final_evaluation.get(key, 0)) < 0:
            raise ValueError(f"final_evaluation.{key} must be nonnegative")

    devices = [str(item) for item in run.get("devices", [])]
    if not devices:
        raise ValueError("run.devices must contain at least one device")
    return value


def training_methods(config: dict) -> list[dict]:
    """Return the three method definitions before LR-sweep expansion."""
    methods = []
    allowed_keys = {
        "name",
        "edit_kind",
        "warmup_fraction",
        "primary",
    }
    for raw in config["variants"]:
        if not isinstance(raw, dict):
            raise ValueError("each training variant must be a mapping")
        extra = sorted(set(raw) - allowed_keys)
        if extra:
            raise ValueError(
                f"variant {raw.get('name', '<unnamed>')} has unknown keys: {extra}"
            )
        method = {
            "primary": False,
            "warmup_fraction": 0.0,
            **raw,
        }
        if not str(method.get("name", "")).strip():
            raise ValueError("every training variant needs a name")
        if str(method.get("edit_kind", "")) not in {
            "fixed_steering",
            "directional_linear",
            "base_mlp",
        }:
            raise ValueError(f"unsupported production edit_kind: {method.get('edit_kind')!r}")
        method["warmup_fraction"] = float(method.get("warmup_fraction", 0.0))
        if not 0.0 <= method["warmup_fraction"] < 1.0:
            raise ValueError(
                f"variant {method['name']} warmup_fraction must be in [0, 1)"
            )
        methods.append(method)
    names = [method["name"] for method in methods]
    if len(names) != len(set(names)):
        raise ValueError("training variant names are not unique")
    return methods


def _learning_rate_tag(value: float) -> str:
    return f"{float(value):.0e}"


def expand_training_variants(config: dict) -> list[dict]:
    """Expand each method over the configured learning-rate sweep."""
    variants = []
    learning_rates = [float(value) for value in config["sweep"]["learning_rates"]]
    for method in training_methods(config):
        for lr in learning_rates:
            variants.append(
                {
                    **method,
                    "method_name": str(method["name"]),
                    "name": f"{method['name']}_lr_{_learning_rate_tag(lr)}",
                    "lr": float(lr),
                }
            )
    return variants


def _argument_name(key: str) -> str:
    return f"--{str(key)}"


def training_command(
    config: dict,
    variant: dict,
    *,
    device: str,
    root: Path,
) -> list[str]:
    training = dict(config["training"])
    output_dir = Path(config["run"]["output_root"]) / variant["name"]
    command = [
        sys.executable,
        str(root / "scripts" / "train_topk_conditional_kl_edit_module.py"),
    ]
    values = {
        **training,
        "lr": float(variant["lr"]),
        "warmup_fraction": float(variant.get("warmup_fraction", 0.0)),
        "output_dir": str(output_dir),
        "device": str(device),
        "edit_kind": variant["edit_kind"],
    }
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                command.append(_argument_name(key))
            continue
        command.extend([_argument_name(key), str(value)])
    return command


def _config_hash(config: dict) -> str:
    payload = json.dumps(
        config, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _run_request(config: dict, variant: dict) -> dict:
    value = {
        "cache_identity": cache_identity(config["training"]["cache_dir"]),
        "training": config["training"],
        "variant": variant,
    }
    return {**value, "request_hash": _config_hash(value)}


def select_best_training_runs(
    results: list[dict],
    *,
    methods: list[dict],
    output_root: Path,
) -> list[dict]:
    """Select one run per method by validation recovery only."""
    selected = []
    for method in methods:
        method_name = str(method["name"])
        candidates = []
        for row in results:
            if str(row.get("method_name", "")) != method_name:
                continue
            recovery = row.get("val_recovery")
            if recovery is None:
                continue
            try:
                recovery_value = float(recovery)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(recovery_value):
                continue
            loss = row.get("val_loss")
            try:
                loss_value = float(loss) if loss is not None else float("inf")
            except (TypeError, ValueError):
                loss_value = float("inf")
            candidates.append((row, recovery_value, loss_value))
        if not candidates:
            raise RuntimeError(
                f"no finite validation recovery is available for method {method_name}"
            )
        # Primary key is maximum validation recovery.  Validation KL and LR are
        # deterministic tie-breakers only; model selection is by recovery.
        best, recovery_value, loss_value = sorted(
            candidates,
            key=lambda item: (
                -item[1],
                item[2],
                float(item[0].get("lr", float("inf"))),
                str(item[0].get("name", "")),
            ),
        )[0]
        alias_dir = output_root / "selected" / method_name
        selected.append(
            {
                "method_name": method_name,
                "edit_kind": str(method["edit_kind"]),
                "run_name": str(best["name"]),
                "lr": float(best["lr"]),
                "val_recovery": recovery_value,
                "val_loss": loss_value if math.isfinite(loss_value) else None,
                "output_dir": str(best["output_dir"]),
                "selected_dir": str(alias_dir),
                "checkpoint": str(alias_dir / "model.pt"),
            }
        )
    return selected


def _materialize_selected_runs(output_root: Path, selected: list[dict]) -> None:
    """Create stable symlink aliases for the validation-selected runs."""
    selected_root = output_root / "selected"
    selected_root.mkdir(parents=True, exist_ok=True)
    expected = {str(row["method_name"]) for row in selected}
    for child in selected_root.iterdir():
        if child.name in expected:
            continue
        if child.is_symlink() or child.is_file():
            child.unlink()
        else:
            shutil.rmtree(child)
    for row in selected:
        alias = selected_root / str(row["method_name"])
        if alias.is_symlink() or alias.is_file():
            alias.unlink()
        elif alias.exists():
            shutil.rmtree(alias)
        target = Path("..") / str(row["run_name"])
        alias.symlink_to(target, target_is_directory=True)


def run_training_sweep(
    config: dict,
    *,
    root: Path,
    devices: list[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    output_root = Path(config["run"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    variants = expand_training_variants(config)
    selected_devices = [
        str(item)
        for item in (
            devices if devices is not None else config["run"]["devices"]
        )
    ]
    if not selected_devices:
        raise ValueError("at least one training device is required")
    progress_positions = {
        device: index for index, device in enumerate(selected_devices)
    }
    queue: Queue = Queue()
    for variant in variants:
        queue.put(variant)
    results = []
    result_lock = threading.Lock()

    def worker(device: str) -> None:
        while True:
            try:
                variant = queue.get_nowait()
            except Empty:
                return
            output_dir = output_root / variant["name"]
            metrics_path = output_dir / "metrics.json"
            checkpoint_path = output_dir / "model.pt"
            history_paths = (
                output_dir / "train_history.json",
                output_dir / "eval_history.json",
            )
            request_path = output_dir / "run_request.json"
            request = _run_request(config, variant)
            existing_request = {}
            if request_path.exists():
                existing_request = json.loads(
                    request_path.read_text(encoding="utf-8")
                )
            command = training_command(
                config, variant, device=device, root=root
            )
            started = time.time()
            if (
                bool(config["run"].get("resume", True))
                and not force
                and metrics_path.exists()
                and checkpoint_path.exists()
                and all(path.exists() for path in history_paths)
                and existing_request.get("request_hash")
                == request["request_hash"]
                # Requests written before completion tracking remain reusable.
                # New attempts must explicitly complete before their artifacts
                # can be accepted (old files may survive a failed retraining).
                and existing_request.get("status") == "complete"
            ):
                status = "reused"
                return_code = 0
            elif dry_run:
                status = "dry_run"
                return_code = 0
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
                _write_json(request_path, {**request, "status": "running"})
                print(
                    f"[training] {variant['name']}: start on {device}",
                    flush=True,
                )
                with (output_dir / "train.log").open(
                    "w", encoding="utf-8"
                ) as handle:
                    process = subprocess.Popen(
                        command,
                        cwd=root,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    assert process.stdout is not None
                    progress = None
                    progress_epoch = None
                    try:
                        for child_line in process.stdout:
                            handle.write(child_line)
                            handle.flush()
                            parsed = _structured_child_event(child_line)
                            if (
                                parsed is not None
                                and parsed[0]
                                == "topk_kl_distill_train_microbatch"
                            ):
                                payload = parsed[1]
                                epoch = int(payload.get("epoch", 0))
                                batch = int(payload.get("batch", 0))
                                batches = int(payload.get("batches", 0))
                                if progress is None or progress_epoch != epoch:
                                    if progress is not None:
                                        progress.close()
                                    progress = tqdm(
                                        total=max(1, batches),
                                        desc=(
                                            f"[training][{variant['name']}] "
                                            f"epoch {epoch}"
                                        ),
                                        unit="microbatch",
                                        dynamic_ncols=True,
                                        position=progress_positions[device],
                                        leave=True,
                                        disable=not sys.stdout.isatty(),
                                        file=sys.stdout,
                                    )
                                    progress_epoch = epoch
                                progress.update(max(0, batch - progress.n))
                                progress.set_postfix(
                                    update=payload.get("update_step", "?"),
                                    accum=(
                                        f"{payload.get('microbatches_in_update', '?')}"
                                        f"/{payload.get('microbatches_per_update', '?')}"
                                    ),
                                    loss=f"{float(payload.get('loss', float('nan'))):.6f}",
                                    recovery=(
                                        f"{float(payload.get('recovery', float('nan'))):.6f}"
                                    ),
                                    refresh=True,
                                )
                                if batch >= batches > 0:
                                    progress.close()
                                    progress = None
                                    progress_epoch = None
                                continue
                            if (
                                parsed is not None
                                and parsed[0]
                                == "topk_kl_distill_train_update"
                            ):
                                # The bar already displays update, loss, and
                                # recovery. Preserve the richer event in
                                # train.log without adding terminal lines.
                                continue
                            live = _format_live_child_line(child_line)
                            if live:
                                message = (
                                    f"[training][{variant['name']}] {live}"
                                )
                                if progress is not None:
                                    progress.write(message, file=sys.stdout)
                                else:
                                    print(message, flush=True)
                        return_code = int(process.wait())
                    finally:
                        if progress is not None:
                            progress.close()
                status = "complete" if return_code == 0 else "failed"
                _write_json(request_path, {**request, "status": status})
                print(
                    f"[training] {variant['name']}: {status} "
                    f"on {device} ({time.time() - started:.1f}s)",
                    flush=True,
                )
            metrics = {}
            if metrics_path.exists():
                metrics = json.loads(
                    metrics_path.read_text(encoding="utf-8")
                )
            row = {
                **variant,
                "device": device,
                "status": status,
                "return_code": return_code,
                "elapsed_seconds": time.time() - started,
                "output_dir": str(output_dir),
                "command": command,
                "request_hash": request["request_hash"],
                "trainable_parameters": metrics.get(
                    "trainable_parameters"
                ),
                "val_loss": metrics.get("val_edited_kl_mean"),
                "val_recovery": metrics.get("val_recovery"),
                "selected_checkpoint_epoch": metrics.get(
                    "selected_checkpoint_epoch"
                ),
                "validation_source_splits": metrics.get(
                    "validation_source_splits", ["val"]
                ),
            }
            with result_lock:
                results.append(row)
            queue.task_done()

    threads = [
        threading.Thread(target=worker, args=(device,), daemon=False)
        for device in selected_devices
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    results.sort(key=lambda row: (str(row.get("method_name", "")), float(row.get("lr", 0.0))))
    failed = [row["name"] for row in results if row["status"] == "failed"]
    if failed:
        manifest = {
            "config_hash": _config_hash(config),
            "devices": selected_devices,
            "force": bool(force),
            "dry_run": bool(dry_run),
            "selection_metric": "val_recovery",
            "selected": [],
            "runs": results,
        }
        _write_json(output_root / "training_manifest.json", manifest)
        raise RuntimeError(
            "training variants failed; inspect their train.log files: "
            + ", ".join(failed)
        )

    selected = [] if dry_run else select_best_training_runs(
        results,
        methods=training_methods(config),
        output_root=output_root,
    )
    if not dry_run:
        _materialize_selected_runs(output_root, selected)
        selection = {
            "selection_metric": "val_recovery",
            "selected": selected,
        }
        _write_json(output_root / "training_selection.json", selection)
        for row in selected:
            print(
                f"[training] selected {row['method_name']}: "
                f"lr={float(row['lr']):.3e} "
                f"val_recovery={float(row['val_recovery']):.6f}",
                flush=True,
            )

    manifest = {
        "config_hash": _config_hash(config),
        "devices": selected_devices,
        "force": bool(force),
        "dry_run": bool(dry_run),
        "selection_metric": "val_recovery",
        "selected": selected,
        "runs": results,
    }
    _write_json(output_root / "training_manifest.json", manifest)
    return manifest
