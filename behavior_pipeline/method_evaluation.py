from __future__ import annotations

"""Evaluate configured best-validation edit checkpoints on the test split."""

import json
from pathlib import Path

from torch.utils.data import DataLoader  # noqa: E402

from behavior_pipeline.counterfactual_runtime import load_edit_checkpoint  # noqa: E402
from behavior_pipeline.artifact_identity import checkpoint_identity  # noqa: E402
from behavior_pipeline.final_evaluation import (  # noqa: E402
    source_clustered_recovery_interval,
)
from behavior_pipeline.model_utils import resolve_hidden_size  # noqa: E402
from behavior_pipeline.topk_kl import ALL_TOKENS_POSITION_POLICY
from behavior_pipeline.training_harness import (  # noqa: E402
    training_methods,
)
from scripts.train_topk_conditional_kl_edit_module import (  # noqa: E402
    _collate_rows,
    evaluate,
)


def evaluate_methods(config: dict, *, loaded, dataset, manifest: dict,
                     device: str, preloaded: dict | None = None) -> dict:
    """Freshly evaluate every selected behavior editor on the supplied population.

    `preloaded` maps method names to (module, config, metrics, identity), letting
    the objective comparison use the exact same MLP instance for both metrics.
    Return report-ready metrics computed from the supplied evaluation population.
    """
    training = config["training"]
    loader = DataLoader(
        dataset,
        batch_size=int(training.get("eval_batch_size", 4)),
        shuffle=False,
        collate_fn=_collate_rows,
    )
    pad_token_id = int(loaded.tokenizer.pad_token_id or 0)
    rows = []
    output_root = Path(config["run"]["output_root"])
    selection_path = output_root / "training_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"missing validation selection manifest: {selection_path}; run training first"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if str(selection.get("selection_metric", "")) != "val_recovery":
        raise RuntimeError("training selection must use val_recovery")
    selected_by_method = {
        str(row["method_name"]): row for row in selection.get("selected", [])
    }
    methods = training_methods(config)
    for method in methods:
        method_name = str(method["name"])
        selected = selected_by_method.get(method_name)
        if selected is None:
            raise RuntimeError(f"missing selected run for method {method_name}")
        checkpoint = output_root / "selected" / method_name / "model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing selected checkpoint: {checkpoint}")
        if method_name in (preloaded or {}):
            module, checkpoint_config, checkpoint_metrics, identity = preloaded[method_name]
        else:
            identity = checkpoint_identity(checkpoint)
            module, checkpoint_config, checkpoint_metrics = load_edit_checkpoint(
                checkpoint,
                d_model=resolve_hidden_size(loaded.model),
                device=device,
                expected_model_name=str(training["model_name"]),
                expected_layer=int(training.get("layer", 20)),
            )
        common = {
            "model_lm": loaded.model,
            "layer_module": loaded.layers[int(training.get("layer", 20))],
            "edit_module": module,
            "loader": loader,
            "device": device,
            "forward_batch_size": int(training.get("forward_batch_size", 2)),
            "pad_token_id": pad_token_id,
            "edit_position_policy": ALL_TOKENS_POSITION_POLICY,
        }
        combined = evaluate(
            directions="both", include_sequence_rows=True, **common
        )
        forward = evaluate(directions="c_to_i", **common)
        reverse = evaluate(directions="i_to_c", **common)
        uncertainty = source_clustered_recovery_interval(
            list(combined.get("sequence_rows", [])),
            seed=int(training.get("seed", 0)),
        )
        if checkpoint_identity(checkpoint) != identity:
            raise RuntimeError(f"checkpoint changed during evaluation: {checkpoint}")
        rows.append(
            {
                "method": method_name,
                "selected_lr": float(selected["lr"]),
                "validation_recovery": float(selected["val_recovery"]),
                "parameters": int(
                    sum(parameter.numel() for parameter in module.parameters())
                ),
                "test_source_kl_mean": float(combined["source_kl_mean"]),
                "test_edited_kl_mean": float(combined["edited_kl_mean"]),
                "test_recovery": float(combined["recovery"]),
                **uncertainty,
                "c_to_i_recovery": float(forward["recovery"]),
                "i_to_c_recovery": float(reverse["recovery"]),
                "student_support_mass_p10": float(
                    combined["support_mass_p10"]
                ),
                "student_support_mass_min": float(
                    combined["support_mass_min"]
                ),
                "student_support_mass_histogram": dict(
                    combined["support_mass_histogram"]
                ),
                "best_checkpoint_epoch": int(
                    checkpoint_metrics.get(
                        "selected_checkpoint_epoch",
                        checkpoint_config.get("selected_checkpoint_epoch", 0),
                    )
                ),
                "checkpoint_identity": identity,
                "behavior_edit_position_policy": ALL_TOKENS_POSITION_POLICY,
            }
        )
    return {
        "evaluation_manifest_id": manifest["manifest_id"],
        "evaluation_split": "test",
        "metric_unit": "mean conditional KL per response token",
        "scope": manifest["scope"],
        "source_prompts": manifest["source_prompts"],
        "sequence_direction_records": manifest["sequence_direction_records"],
        "response_token_cap": manifest["response_token_cap"],
        "methods": rows,
    }
