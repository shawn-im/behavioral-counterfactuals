#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate selected behavior/activation editors and write both test comparisons."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from behavior_pipeline.runtime_env import configure_runtime_caches  # noqa: E402
configure_runtime_caches(Path(__file__).resolve().parents[1])

from torch.utils.data import DataLoader  # noqa: E402

from behavior_pipeline.activation_reconstruction import ACTIVATION_NMSE_DEFINITION, ActivationReconstructionDataset, row_identity
from behavior_pipeline.activation_training_harness import load_activation_training_config  # noqa: E402
from behavior_pipeline.artifact_identity import checkpoint_identity  # noqa: E402
from behavior_pipeline.final_evaluation import build_evaluation_manifest  # noqa: E402
from behavior_pipeline.method_evaluation import evaluate_methods  # noqa: E402
from behavior_pipeline.counterfactual_runtime import load_edit_checkpoint  # noqa: E402
from behavior_pipeline.model_utils import load_causal_lm  # noqa: E402
from behavior_pipeline.topk_kl import (  # noqa: E402
    ALL_TOKENS_POSITION_POLICY,
    FINAL_QUESTION_RESPONSE_POSITION_POLICY,
    TopKKLDistillDataset,
)
from behavior_pipeline.training_harness import load_training_harness_config  # noqa: E402
from scripts.train_activation_nmse_edit_module import evaluate_nmse  # noqa: E402
from scripts.train_topk_conditional_kl_edit_module import _collate_rows, evaluate as evaluate_behavior  # noqa: E402


def _load_selected(path: Path, metric: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing training selection manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("selection_metric")) != metric:
        raise RuntimeError(f"{path} selects by {payload.get('selection_metric')!r}, expected {metric!r}")
    rows = [row for row in payload.get("selected", []) if str(row.get("method_name")) == "edit_module_base_mlp"]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one selected base MLP in {path}")
    return rows[0]


def _require_full_test(config: dict) -> None:
    settings = config.get("final_evaluation", {})
    if any(int(settings.get(key, 0)) != 0 for key in ("max_records", "response_token_cap")):
        raise RuntimeError(
            "unified evaluation requires final_evaluation.max_records=0 and "
            "response_token_cap=0 so both objectives use the full, untruncated test population"
        )


def _assert_same_population(topk_ds: TopKKLDistillDataset, activation_ds: ActivationReconstructionDataset) -> None:
    topk_ids = [row_identity(row) for row in topk_ds.rows]
    activation_ids = [row_identity(row) for row in activation_ds.index]
    if topk_ids != activation_ids:
        raise RuntimeError("top-k KL and activation caches do not contain the exact same ordered test rows")
    for left, right in zip(topk_ds.rows, activation_ds.index):
        if int(left["response_ids"].numel()) != int(right["response_tokens"]):
            raise RuntimeError("activation-cache response length differs from the behavior cache")
        if int(right["activation_tokens"]) != int(right["question_tokens"]) + int(
            right["response_tokens"]
        ):
            raise RuntimeError(
                "activation-cache region is not exactly final-question plus response tokens"
            )


def _validate_activation_cache(metadata: dict, training: dict, manifest: dict) -> None:
    if (str(metadata.get("model_name")) != str(training["model_name"])
            or int(metadata.get("layer", -1)) != int(training.get("layer", 20))):
        raise RuntimeError("activation cache model/layer differs from the evaluation model/layer")
    if metadata.get("position_policy") != FINAL_QUESTION_RESPONSE_POSITION_POLICY:
        raise RuntimeError("activation cache must cover final-question and response tokens")
    if metadata.get("source_topk_cache_metadata_sha256") != manifest["cache_identity"]["metadata_sha256"]:
        raise RuntimeError("activation cache was not built from the current behavior cache metadata")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kl_config", required=True)
    ap.add_argument("--nmse_config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    kl_config = load_training_harness_config(args.kl_config)
    nmse_config = load_activation_training_config(args.nmse_config)
    _require_full_test(kl_config)
    kl_training = kl_config["training"]
    nmse_training = nmse_config["training"]
    if str(kl_training["model_name"]) != str(nmse_training.get("model_name", "")):
        raise RuntimeError("KL and NMSE configs must use the same model")
    if int(kl_training.get("layer", 20)) != int(nmse_training.get("layer", 20)):
        raise RuntimeError("KL and NMSE configs must use the same edit layer")

    kl_root = Path(kl_config["run"]["output_root"])
    nmse_root = Path(nmse_config["run"]["output_root"])
    kl_selected = _load_selected(kl_root / "training_selection.json", "val_recovery")
    nmse_selected = _load_selected(nmse_root / "training_selection.json", "val_nmse")
    kl_checkpoint = kl_root / "selected" / "edit_module_base_mlp" / "model.pt"
    nmse_checkpoint = nmse_root / "selected" / "edit_module_base_mlp" / "model.pt"
    if not kl_checkpoint.is_file() or not nmse_checkpoint.is_file():
        raise FileNotFoundError("selected KL and NMSE MLP checkpoints must both exist")

    behavior_ds = TopKKLDistillDataset(Path(kl_training["cache_dir"]), "test")
    activation_ds = ActivationReconstructionDataset(Path(nmse_training["cache_dir"]), "test")
    _assert_same_population(behavior_ds, activation_ds)
    manifest = build_evaluation_manifest(
        behavior_ds.rows,
        cache_dir=kl_training["cache_dir"],
        model_name=kl_training["model_name"],
        layer=int(kl_training.get("layer", 20)),
        response_token_cap=0,
        maximum=0,
    )
    _validate_activation_cache(activation_ds.meta, kl_training, manifest)
    activation_loader = DataLoader(
        activation_ds,
        batch_size=int(nmse_training.get("eval_batch_size", 4)),
        shuffle=False,
        collate_fn=_collate_rows,
    )

    device = str(args.device or kl_config["run"]["devices"][0])
    d_model = int(activation_ds.meta["d_model"])
    checkpoint_arguments = {
        "d_model": d_model,
        "device": device,
        "expected_model_name": str(kl_training["model_name"]),
        "expected_layer": int(kl_training.get("layer", 20)),
    }
    kl_identity = checkpoint_identity(kl_checkpoint)
    nmse_identity = checkpoint_identity(nmse_checkpoint)
    kl_module, kl_ckpt_config, kl_ckpt_metrics = load_edit_checkpoint(kl_checkpoint, **checkpoint_arguments)
    nmse_module, nmse_ckpt_config, _nmse_ckpt_metrics = load_edit_checkpoint(nmse_checkpoint, **checkpoint_arguments)
    if (
        str(nmse_ckpt_config.get("nmse_definition", "")) != ACTIVATION_NMSE_DEFINITION
        or str(nmse_ckpt_config.get("activation_position_policy", ""))
        != FINAL_QUESTION_RESPONSE_POSITION_POLICY
    ):
        raise RuntimeError(
            "selected activation-NMSE checkpoint has an unsupported normalization or position policy; "
            "rerun the activation-NMSE training sweep"
        )
    kl_activation = evaluate_nmse(kl_module, activation_loader, device=device, directions="both")
    nmse_activation = evaluate_nmse(nmse_module, activation_loader, device=device, directions="both")

    # Both comparisons come from this invocation, using one loaded base model
    # and the same in-memory KL MLP used for activation evaluation above.
    loaded = load_causal_lm(str(kl_training["model_name"]), device=device, dtype=str(kl_training.get("dtype", "bf16")))
    for parameter in loaded.model.parameters():
        parameter.requires_grad_(False)
    method_comparison = evaluate_methods(
        kl_config, loaded=loaded, dataset=behavior_ds, manifest=manifest, device=device,
        preloaded={"edit_module_base_mlp": (kl_module, kl_ckpt_config, kl_ckpt_metrics, kl_identity)},
    )
    kl_rows = [row for row in method_comparison["methods"] if row["method"] == "edit_module_base_mlp"]
    if len(kl_rows) != 1:
        raise RuntimeError("unified evaluation requires exactly one configured behavior base MLP")
    kl_behavior = kl_rows[0]
    behavior_loader = DataLoader(
        behavior_ds,
        batch_size=int(kl_training.get("eval_batch_size", 4)),
        shuffle=False,
        collate_fn=_collate_rows,
    )
    nmse_behavior = evaluate_behavior(
        loaded.model,
        loaded.layers[int(kl_training.get("layer", 20))],
        nmse_module,
        behavior_loader,
        device=device,
        directions="both",
        forward_batch_size=int(kl_training.get("forward_batch_size", 2)),
        pad_token_id=int(loaded.tokenizer.pad_token_id or 0),
        tokenizer=loaded.tokenizer,
        edit_position_policy=FINAL_QUESTION_RESPONSE_POSITION_POLICY,
    )

    rows = [
        {
            "objective": "behavior_kl",
            "label": "Behavior reconstruction (top-k KL)",
            "selection_metric": "val_recovery",
            "selected_lr": float(kl_selected["lr"]),
            "validation_metric": float(kl_selected["val_recovery"]),
            "test_behavior_source_kl": float(kl_behavior["test_source_kl_mean"]),
            "test_behavior_edited_kl": float(kl_behavior["test_edited_kl_mean"]),
            "test_behavior_recovery": float(kl_behavior["test_recovery"]),
            "test_activation_source_nmse": float(kl_activation["source_nmse"]),
            "test_activation_nmse": float(kl_activation["nmse"]),
            "test_activation_recovery": float(kl_activation["activation_recovery"]),
            "behavior_edit_position_policy": ALL_TOKENS_POSITION_POLICY,
            "checkpoint_identity": kl_identity,
        },
        {
            "objective": "activation_nmse",
            "label": "Activation reconstruction (NMSE)",
            "selection_metric": "val_nmse",
            "selected_lr": float(nmse_selected["lr"]),
            "validation_metric": float(nmse_selected["val_nmse"]),
            "test_behavior_source_kl": float(nmse_behavior["source_kl_mean"]),
            "test_behavior_edited_kl": float(nmse_behavior["edited_kl_mean"]),
            "test_behavior_recovery": float(nmse_behavior["recovery"]),
            "test_activation_source_nmse": float(nmse_activation["source_nmse"]),
            "test_activation_nmse": float(nmse_activation["nmse"]),
            "test_activation_recovery": float(nmse_activation["activation_recovery"]),
            "behavior_edit_position_policy": FINAL_QUESTION_RESPONSE_POSITION_POLICY,
            "checkpoint_identity": nmse_identity,
        },
    ]
    default_output = kl_root.parent / "reconstruction_objective_comparison" / "test_comparison.json"
    output = Path(args.output or default_output)
    payload = {
        "evaluation_manifest_id": manifest["manifest_id"],
        "population": "same fixed-prefix test rows for both objectives",
        "nmse_definition": ACTIVATION_NMSE_DEFINITION,
        "activation_position_policy": str(activation_ds.meta.get("position_policy")),
        "activation_editor_behavior_position_policy": FINAL_QUESTION_RESPONSE_POSITION_POLICY,
        "source_prompts": len({str(row["source_id"]) for row in behavior_ds.rows}),
        "sequence_rows": len(behavior_ds.rows),
        "objectives": rows,
    }
    for checkpoint, identity in ((kl_checkpoint, kl_identity), (nmse_checkpoint, nmse_identity)):
        if checkpoint_identity(checkpoint) != identity:
            raise RuntimeError(f"checkpoint changed during evaluation: {checkpoint}")
    final_dir = kl_root / "final_evaluation"
    method_output = final_dir / "test_comparison.json"
    manifest_output = final_dir / "evaluation_manifest.json"
    if output.resolve() in {method_output.resolve(), manifest_output.resolve()}:
        raise ValueError("--output must not overwrite the method comparison or evaluation manifest")
    # Publish only after every evaluation and consistency check succeeds.
    final_dir.mkdir(parents=True, exist_ok=True)
    method_output.write_text(json.dumps(method_comparison, indent=2), encoding="utf-8")
    manifest_output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"objective_comparison": str(output), "method_comparison": str(method_output),
                      "manifest": str(manifest_output), "rows": len(rows)}))


if __name__ == "__main__":
    main()
