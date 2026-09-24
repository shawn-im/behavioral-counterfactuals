from __future__ import annotations

"""Internal core implementation for :mod:`behavior_pipeline.counterfactual_runtime`."""

import inspect
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .edit_modules import BaseMLPEdit, DirectionalLinearEdit, SteeringVectorEdit, make_edit_module, set_edit_input_norm


def json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(obj).__name__}")


def write_json(path: Path, value) -> None:
    """Write compact ordinary JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, separators=(",", ":"), default=json_default),
        encoding="utf-8",
    )


def read_json(path: Path, default=None):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def _signed_phrase_spans(
    record: dict,
    values: torch.Tensor,
    *,
    max_span_tokens: int,
    top_n: int,
    deduplicate_overlap: bool = True,
) -> dict[str, list[dict]]:
    """Extract high-scoring short same-sign spans without Python O(T * span) scans.

    Evaluate candidate windows with vectorized tensor views and
    only materialize a bounded set of high-scoring spans as Python objects.
    """
    values = values.detach().float().cpu().reshape(-1)
    tokens = list(record.get("tokens", []))[: int(values.numel())]
    n = min(len(tokens), int(values.numel()))
    if n <= 0 or int(top_n) <= 0 or int(max_span_tokens) <= 0:
        return {"positive": [], "negative": []}
    values = values[:n]
    regions = torch.tensor(
        [1 if bool(token.get("is_prompt", False)) else 0 for token in tokens[:n]],
        dtype=torch.int8,
    )
    candidate_limit = max(64, int(top_n) * max(1, int(max_span_tokens)) * 8)
    candidates: dict[int, list[tuple[float, int, int]]] = {1: [], -1: []}

    for length in range(1, min(int(max_span_tokens), n) + 1):
        windows = values.unfold(0, length, 1)
        signs = torch.sign(windows)
        first_sign = signs[:, 0]
        valid = first_sign.ne(0) & signs.eq(first_sign.unsqueeze(1)).all(dim=1)
        region_windows = regions.unfold(0, length, 1)
        valid &= region_windows.eq(region_windows[:, :1]).all(dim=1)
        sums = windows.sum(dim=1)
        for sign in (1, -1):
            mask = valid & (first_sign == sign)
            indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
            if int(indices.numel()) == 0:
                continue
            scores = sums[indices].abs()
            keep = min(candidate_limit, int(indices.numel()))
            top_offsets = torch.topk(scores, k=keep, largest=True, sorted=False).indices
            starts = indices[top_offsets]
            for start_index in starts.tolist():
                score = float(sums[start_index])
                candidates[sign].append((abs(score), int(start_index), int(length)))

    def select(sign: int) -> list[dict]:
        ranked = sorted(candidates[sign], key=lambda item: item[0], reverse=True)
        selected: list[dict] = []
        occupied: set[int] = set()
        seen: set[tuple[int, int]] = set()
        for _magnitude, start, length in ranked:
            key = (start, length)
            if key in seen:
                continue
            seen.add(key)
            end = start + length
            indices = set(range(start, end))
            if deduplicate_overlap and occupied & indices:
                continue
            phrase = "".join(
                str(token.get("token_text", "")) for token in tokens[start:end]
            ).strip()
            phrase = " ".join(phrase.split())
            if not phrase:
                continue
            span_values = values[start:end]
            selected.append(
                {
                    "phrase": phrase[:240],
                    "start_token_index": int(start),
                    "end_token_index": int(end - 1),
                    "is_prompt": bool(tokens[start].get("is_prompt", False)),
                    "score_sum": float(span_values.sum()),
                    "score_mean": float(span_values.mean()),
                    "length": int(length),
                }
            )
            occupied.update(indices)
            if len(selected) >= int(top_n):
                break
        return selected

    return {"positive": select(1), "negative": select(-1)}


def is_static_mlp(module: nn.Module) -> bool:
    return isinstance(module, (BaseMLPEdit, DirectionalLinearEdit, SteeringVectorEdit))


def _direction_names_from_signs(direction: torch.Tensor) -> torch.Tensor:
    return direction.reshape(-1) > 0


def decode_applied_edit(
    module: nn.Module,
    features: torch.Tensor,
    direction: str | float | torch.Tensor,
) -> torch.Tensor:
    """Decode editor features into the residual edit applied by a direction."""
    if isinstance(direction, str):
        if direction not in {"c_to_i", "i_to_c"}:
            raise ValueError(f"unknown direction: {direction}")
        signs = torch.tensor(
            [1.0 if direction == "c_to_i" else -1.0],
            device=features.device,
        )
    else:
        signs = torch.as_tensor(direction, device=features.device).reshape(-1)
    if signs.numel() == 1:
        signs = signs.expand(int(features.shape[0]))
    if signs.numel() != int(features.shape[0]):
        raise ValueError("direction count must equal feature batch size")
    if isinstance(module, BaseMLPEdit):
        return module.decode_features(features.float(), signs).float()
    if isinstance(module, DirectionalLinearEdit):
        forward = module.output_heads["c_to_i"](features.float())
        reverse = module.output_heads["i_to_c"](features.float())
        return torch.where(
            _direction_names_from_signs(signs).view(-1, 1, 1),
            forward,
            reverse,
        ).float()
    if isinstance(module, SteeringVectorEdit):
        return features.float() * signs.view(-1, 1, 1).float()
    raise TypeError(f"Unsupported production edit module: {type(module).__name__}")


def load_edit_checkpoint(
    checkpoint: str | Path,
    *,
    d_model: int,
    device: str,
    expected_model_name: str | None = None,
    expected_layer: int | None = None,
) -> tuple[nn.Module, dict, dict]:
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu")
    config = dict(payload.get("config", {}))
    metrics = dict(payload.get("metrics", {}))
    for key, expected in (("model_name", expected_model_name), ("layer", expected_layer)):
        if expected is not None and config.get(key) != expected:
            raise RuntimeError(
                f"checkpoint {key} mismatch: trained with {config.get(key)!r}, "
                f"requested {expected!r}; use the matching checkpoint or configuration"
            )

    def checkpoint_value(key: str, default):
        value = config.get(key)
        if value is None:
            value = metrics.get(key)
        return default if value is None else value

    kind = str(checkpoint_value("edit_kind", "base_mlp"))
    if kind not in {"fixed_steering", "directional_linear", "base_mlp"}:
        raise RuntimeError(
            f"unsupported production edit_kind={kind!r}; retrain with the current code"
        )
    module_kwargs = {}
    if kind == "base_mlp":
        module_kwargs = {
            "intermediate_size": int(checkpoint_value("edit_mlp_hidden_size", d_model)),
            "activation_name": str(checkpoint_value("edit_mlp_activation", "silu")),
            "gated_mlp": bool(checkpoint_value("edit_mlp_gated", True)),
        }
    module = make_edit_module(kind, int(d_model), **module_kwargs)
    state_dict = payload["state_dict"]
    module.load_state_dict(state_dict, strict=True)
    module.to(device)
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    set_edit_input_norm(module, str(config.get("input_norm_mode", "rms")))
    return module, config, metrics


@dataclass
class Runtime:
    model: nn.Module
    tokenizer: object
    layer_module: nn.Module
    edit_module: nn.Module
    checkpoint_config: dict
    checkpoint_metrics: dict
    device: str
    final_norm: nn.Module | None = None
    output_embeddings: nn.Module | None = None
    model_forward_parameters: frozenset[str] = frozenset()
    last_token_logits_kwargs: dict | None = None


def load_runtime(config: dict, *, device_override: str | None = None) -> Runtime:
    from .model_utils import (
        load_causal_lm,
        resolve_final_norm,
        resolve_hidden_size,
    )

    device = str(device_override or config["model"]["device"])
    loaded = load_causal_lm(
        str(config["model"]["name"]),
        device=device,
        dtype=str(config["model"]["dtype"]),
    )
    model = loaded.model
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layer_index = int(config["model"]["layer"])
    layer_module = loaded.layers[layer_index]
    final_norm = resolve_final_norm(model, loaded.layers)
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise AttributeError(
            f"{type(model).__name__} does not expose output embeddings"
        )
    edit_module, checkpoint_config, checkpoint_metrics = load_edit_checkpoint(
        config["edit"]["checkpoint"],
        d_model=resolve_hidden_size(model),
        device=device,
        expected_model_name=str(config["model"]["name"]),
        expected_layer=layer_index,
    )
    if not is_static_mlp(edit_module):
        raise ValueError(
            "The counterfactual harness requires a current static edit checkpoint; "
            f"got {checkpoint_config.get('edit_kind')!r}"
        )
    return Runtime(
        model=model,
        tokenizer=loaded.tokenizer,
        layer_module=layer_module,
        edit_module=edit_module,
        checkpoint_config=checkpoint_config,
        checkpoint_metrics=checkpoint_metrics,
        device=device,
        final_norm=final_norm,
        output_embeddings=output_embeddings,
        model_forward_parameters=frozenset(_model_forward_parameters(model)),
        last_token_logits_kwargs=_last_token_logits_kwarg(model),
    )


class _CaptureHook:
    def __init__(self, layer_module):
        self.layer_module = layer_module
        self.handle = None
        self.hidden = None

    def _hook(self, _module, _inputs, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        self.hidden = hidden.detach()

    def __enter__(self):
        self.handle = self.layer_module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()


class _FixedEditHook:
    def __init__(self, layer_module, edit: torch.Tensor):
        self.layer_module = layer_module
        self.edit = edit
        self.handle = None

    def _hook(self, _module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        is_list = isinstance(output, list)
        hidden = output[0] if (is_tuple or is_list) else output
        patched = hidden + self.edit.to(device=hidden.device, dtype=hidden.dtype)
        if is_tuple:
            return (patched, *output[1:])
        if is_list:
            result = list(output)
            result[0] = patched
            return result
        return patched

    def __enter__(self):
        self.handle = self.layer_module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()


class _InlineDirectionalEditHook:
    """Compute and apply a directional edit in one language-model forward pass.

    The edit is a deterministic function of the edit-layer output. The hook
    computes that edit and returns the patched activation to downstream layers.
    """

    def __init__(self, layer_module, edit_module: nn.Module, directions: torch.Tensor):
        self.layer_module = layer_module
        self.edit_module = edit_module
        self.directions = directions.detach().float().cpu()
        self.handle = None
        self.edit = None

    def _hook(self, _module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        is_list = isinstance(output, list)
        hidden = output[0] if (is_tuple or is_list) else output
        directions = self.directions.to(device=hidden.device, dtype=torch.float32)
        if hasattr(self.edit_module, "forward_direction"):
            edit = self.edit_module.forward_direction(hidden, directions)
        elif isinstance(self.edit_module, SteeringVectorEdit):
            base = self.edit_module(hidden)
            edit = base * directions.to(base.dtype).view(-1, 1, 1)
        else:
            raise TypeError(
                f"inline tracing does not support {type(self.edit_module).__name__}"
            )
        # forward_direction already returns the model hidden dtype (bf16 in the
        # production analysis runs).  Keep that exact value instead of inflating
        # it to fp32; downstream attribution explicitly casts to fp32 when needed.
        self.edit = edit.detach()
        patched = hidden + edit.to(device=hidden.device, dtype=hidden.dtype)
        if is_tuple:
            return (patched, *output[1:])
        if is_list:
            result = list(output)
            result[0] = patched
            return result
        return patched

    def __enter__(self):
        self.handle = self.layer_module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@torch.no_grad()
def capture_hidden(
    runtime: Runtime,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    with _CaptureHook(runtime.layer_module) as hook:
        runtime.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    if hook.hidden is None:
        raise RuntimeError("Edit-layer hidden-state capture failed")
    return hook.hidden.detach().float()


def trace_from_hidden(module: nn.Module, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
    if isinstance(module, BaseMLPEdit):
        features = module.features(hidden).float()
    elif isinstance(module, DirectionalLinearEdit):
        features = module.preprocess_input(hidden).float()
    elif isinstance(module, SteeringVectorEdit):
        features = module.v.view(1, 1, -1).expand_as(hidden).float()
    else:
        raise TypeError(f"Unsupported production edit module: {type(module).__name__}")
    errors = []
    edits = {}
    for direction, sign in (("c_to_i", 1.0), ("i_to_c", -1.0)):
        edit = decode_applied_edit(module, features, direction)
        edits[direction] = edit
        if hasattr(module, "forward_direction"):
            direct = module.forward_direction(hidden.float(), sign).float()
        else:
            direct = module(hidden.float()).float() * sign
        errors.append(
            float(
                (
                    torch.linalg.vector_norm(edit - direct)
                    / torch.linalg.vector_norm(direct).clamp_min(1e-12)
                ).detach()
            )
        )
    error = max(errors)
    if error > 1e-5:
        raise RuntimeError(
            "Checkpoint does not match the required feedthrough recurrence: "
            f"trace reconstruction error={error:.3e}"
        )
    return {
        "edit_by_direction": edits,
        "edit_reconstruction_error": torch.tensor(error),
    }


def select_diverse_rows(rows: list[dict], maximum: int) -> list[dict]:
    if int(maximum) <= 0 or len(rows) <= int(maximum):
        return list(rows)
    by_source: dict[str, list[dict]] = {}
    for index, row in enumerate(rows):
        by_source.setdefault(str(row.get("source_id", index)), []).append(row)
    selected = []
    depth = 0
    while len(selected) < int(maximum):
        added = False
        for key in sorted(by_source):
            values = by_source[key]
            if depth < len(values):
                selected.append(values[depth])
                added = True
                if len(selected) >= int(maximum):
                    break
        if not added:
            break
        depth += 1
    return selected


def truncate_cache_row(row: dict, response_cap: int) -> dict:
    cap = int(response_cap)
    response_length = int(row["response_ids"].numel())
    if cap <= 0 or response_length <= cap:
        return row
    result = dict(row)
    response_keys = (
        "response_ids",
        "support_ids",
        "support_mask",
        "logp_control",
        "logp_intervention",
        "logp_control_other",
        "logp_intervention_other",
    )
    for key in response_keys:
        value = result.get(key)
        if isinstance(value, torch.Tensor) and value.ndim >= 1:
            result[key] = value[:cap]
    result["analysis_truncated_from_response_tokens"] = response_length
    return result


def direction_sign(direction: str) -> float:
    if str(direction) == "c_to_i":
        return 1.0
    if str(direction) == "i_to_c":
        return -1.0
    raise ValueError(f"Unknown direction: {direction}")


def target_prompt_ids(row: dict, direction: str) -> torch.Tensor:
    return (
        row["intervention_prompt_ids"]
        if str(direction) == "c_to_i"
        else row["control_prompt_ids"]
    )


def _model_forward_parameters(model: nn.Module) -> set[str]:
    try:
        return set(inspect.signature(model.forward).parameters)
    except (TypeError, ValueError):
        return set()


def _last_token_logits_kwarg(model: nn.Module) -> dict:
    """Use the model-family native last-logit shortcut when it is exposed."""
    parameters = _model_forward_parameters(model)
    if "logits_to_keep" in parameters:
        return {"logits_to_keep": 1}
    if "num_logits_to_keep" in parameters:
        return {"num_logits_to_keep": 1}
    return {}
