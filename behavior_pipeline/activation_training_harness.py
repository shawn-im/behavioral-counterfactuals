from __future__ import annotations

"""Learning-rate sweep harness for activation-NMSE residual editors."""

import hashlib
import json
import math
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

from .activation_reconstruction import ACTIVATION_NMSE_DEFINITION, ensure_activation_cache_indices
from .artifact_identity import cache_identity


def load_activation_training_config(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("activation training config must be a YAML mapping")
    for section in ("run", "training", "sweep"):
        if not isinstance(value.get(section), dict):
            raise ValueError(f"missing mapping section: {section}")
    run = value["run"]
    training = value["training"]
    sweep = value["sweep"]
    if not str(run.get("output_root", "")).strip():
        raise ValueError("run.output_root must be set")
    devices = [str(item) for item in run.get("devices", [])]
    if not devices:
        raise ValueError("run.devices must contain at least one device")
    if not str(training.get("cache_dir", "")).strip():
        raise ValueError("training.cache_dir must be set")
    if str(training.get("input_norm_mode", "rms")) != "rms":
        raise ValueError("activation-NMSE comparison uses the same RMS input normalization as the KL MLP")
    if str(training.get("directions", "both")) != "both":
        raise ValueError("activation-NMSE comparison must train both directions")
    if int(training.get("epochs", 1)) <= 0:
        raise ValueError("training.epochs must be positive")
    rates = [float(item) for item in sweep.get("learning_rates", [])]
    if not rates or any((not math.isfinite(rate) or rate <= 0.0) for rate in rates):
        raise ValueError("sweep.learning_rates must contain finite positive values")
    if len(rates) != len(set(rates)):
        raise ValueError("sweep.learning_rates must not contain duplicates")
    if str(sweep.get("selection_metric", "")) != "val_nmse":
        raise ValueError("sweep.selection_metric must be val_nmse")
    warmup = float(training.get("warmup_fraction", 0.1))
    if not 0.0 <= warmup < 1.0:
        raise ValueError("training.warmup_fraction must be in [0,1)")
    return value


def _tag(lr: float) -> str:
    return f"{float(lr):.0e}"


def _config_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _request(config: dict, lr: float) -> dict:
    payload = {
        "activation_nmse_definition": ACTIVATION_NMSE_DEFINITION,
        "activation_cache_identity": cache_identity(config["training"]["cache_dir"]),
        "training": config["training"],
        "lr": float(lr),
    }
    return {**payload, "request_hash": _config_hash(payload)}


def _command(config: dict, lr: float, *, device: str, root: Path, output_dir: Path) -> list[str]:
    command = [sys.executable, str(root / "scripts" / "train_activation_nmse_edit_module.py")]
    values = {**config["training"], "lr": float(lr), "device": device, "output_dir": str(output_dir)}
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                command.append(f"--{key}")
        else:
            command.extend([f"--{key}", str(value)])
    return command


def run_activation_training_sweep(config: dict, *, root: Path, devices: list[str] | None = None, force: bool = False, dry_run: bool = False) -> dict:
    output_root = Path(config["run"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if not dry_run:
        ensure_activation_cache_indices(
            config["training"]["cache_dir"], progress=True
        )
    selected_devices = [str(item) for item in (devices if devices is not None else config["run"]["devices"])]
    rates = [float(item) for item in config["sweep"]["learning_rates"]]
    jobs: queue.Queue = queue.Queue()
    for lr in rates:
        jobs.put(lr)
    results: list[dict] = []
    lock = threading.Lock()

    def worker(device: str) -> None:
        while True:
            try:
                lr = jobs.get_nowait()
            except queue.Empty:
                return
            name = f"edit_module_base_mlp_lr_{_tag(lr)}"
            out_dir = output_root / name
            request_path = out_dir / "run_request.json"
            metrics_path = out_dir / "metrics.json"
            checkpoint_path = out_dir / "model.pt"
            request = _request(config, lr)
            existing = {}
            if request_path.is_file():
                try:
                    existing = json.loads(request_path.read_text(encoding="utf-8"))
                except Exception:
                    existing = {}
            command = _command(config, lr, device=device, root=root, output_dir=out_dir)
            started = time.time()
            if bool(config["run"].get("resume", True)) and not force and metrics_path.is_file() and checkpoint_path.is_file() and existing.get("request_hash") == request["request_hash"]:
                status = "reused"
                return_code = 0
            elif dry_run:
                status = "dry_run"
                return_code = 0
            else:
                out_dir.mkdir(parents=True, exist_ok=True)
                request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
                print(f"[activation-training] {name}: start on {device}", flush=True)
                with (out_dir / "train.log").open("w", encoding="utf-8") as handle:
                    process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                    assert process.stdout is not None
                    for line in process.stdout:
                        handle.write(line)
                        handle.flush()
                        stripped = line.strip()
                        if "activation_nmse_train_update" in stripped or "activation_nmse_eval" in stripped:
                            print(f"[activation-training][{name}] {stripped}", flush=True)
                    return_code = int(process.wait())
                status = "complete" if return_code == 0 else "failed"
                print(f"[activation-training] {name}: {status} on {device} ({time.time()-started:.1f}s)", flush=True)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
            row = {
                "name": name,
                "method_name": "edit_module_base_mlp",
                "edit_kind": "base_mlp",
                "lr": float(lr),
                "device": device,
                "status": status,
                "return_code": return_code,
                "elapsed_seconds": time.time() - started,
                "output_dir": str(out_dir),
                "command": command,
                "request_hash": request["request_hash"],
                "val_nmse": metrics.get("nmse", metrics.get("val_nmse")),
                "val_activation_recovery": metrics.get("activation_recovery"),
                "trainable_parameters": metrics.get("trainable_parameters"),
            }
            with lock:
                results.append(row)
            jobs.task_done()

    threads = [threading.Thread(target=worker, args=(device,), daemon=False) for device in selected_devices]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    results.sort(key=lambda row: float(row["lr"]))
    failed = [row["name"] for row in results if row["status"] == "failed"]
    if failed:
        manifest = {
            "activation_nmse_definition": ACTIVATION_NMSE_DEFINITION,
            "selection_metric": "val_nmse",
            "selected": [],
            "runs": results,
        }
        (output_root / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise RuntimeError("activation-NMSE runs failed: " + ", ".join(failed))
    selected = []
    if not dry_run:
        candidates = []
        for row in results:
            value = row.get("val_nmse")
            if value is None:
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(parsed):
                candidates.append((parsed, float(row["lr"]), row))
        if not candidates:
            raise RuntimeError("no finite validation NMSE is available for checkpoint selection")
        best_nmse, _lr, best = sorted(candidates, key=lambda item: (item[0], item[1]))[0]
        selected_root = output_root / "selected"
        selected_root.mkdir(parents=True, exist_ok=True)
        alias = selected_root / "edit_module_base_mlp"
        if alias.is_symlink() or alias.is_file():
            alias.unlink()
        elif alias.exists():
            shutil.rmtree(alias)
        alias.symlink_to(Path("..") / str(best["name"]), target_is_directory=True)
        selected = [{
            "method_name": "edit_module_base_mlp",
            "edit_kind": "base_mlp",
            "run_name": str(best["name"]),
            "lr": float(best["lr"]),
            "val_nmse": float(best_nmse),
            "val_activation_recovery": best.get("val_activation_recovery"),
            "output_dir": str(best["output_dir"]),
            "selected_dir": str(alias),
            "checkpoint": str(alias / "model.pt"),
        }]
        selection = {
            "activation_nmse_definition": ACTIVATION_NMSE_DEFINITION,
            "selection_metric": "val_nmse",
            "selected": selected,
        }
        (output_root / "training_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
        print(f"[activation-training] selected edit_module_base_mlp: lr={float(best['lr']):.3e} val_nmse={float(best_nmse):.6f}", flush=True)
    manifest = {
        "activation_nmse_definition": ACTIVATION_NMSE_DEFINITION,
        "selection_metric": "val_nmse",
        "selected": selected,
        "runs": results,
    }
    (output_root / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
