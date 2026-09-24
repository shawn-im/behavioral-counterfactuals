from __future__ import annotations

"""Relevance-patching backward rules for Hugging Face transformer models.

This implements the Gemma-family RelP rule set from Jafari et al. (2025):
LN-rule for RMSNorm, Identity-rule for nonlinear gated-MLP activations,
LRP-0 (ordinary linear backward), Half-rule for multiplicative gates, and no
attention-specific rule.  The same Identity/Half treatment is extended to
Gemma-4's per-layer-input multiplicative gate.  Forward values are preserved
exactly by straight-through constructions; only the backward propagation
coefficient changes.
"""

from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import MethodType

import torch
import torch.nn as nn


@dataclass(frozen=True)
class RelPRuleDiagnostics:
    downstream_layers: int
    rmsnorm_modules: int
    gated_mlp_modules: int
    per_layer_input_gates: int = 0
    rules: tuple[str, ...] = ("LN-rule", "Identity-rule", "0-rule", "Half-rule")
    attention_rule: str = "none"


def _resolve_layers_without_transformers(model: nn.Module):
    candidate_paths = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("decoder", "layers"),
        ("model", "decoder", "layers"),
        ("language_model", "model", "layers"),
        ("layers",),
    ]
    for path in candidate_paths:
        obj = model
        for name in path:
            obj = getattr(obj, name, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    if hasattr(model, "get_decoder"):
        decoder = model.get_decoder()
        if hasattr(decoder, "layers"):
            return decoder.layers
    raise AttributeError(f"Cannot locate transformer layers for {type(model).__name__}")


def _is_rmsnorm(module: nn.Module) -> bool:
    name = type(module).__name__.lower()
    return "rmsnorm" in name or "rms_norm" in name


def _gated_mlp_parts(module: nn.Module):
    gate_proj = getattr(module, "gate_proj", None)
    up_proj = getattr(module, "up_proj", None)
    down_proj = getattr(module, "down_proj", None)
    act_fn = getattr(module, "act_fn", None)
    if act_fn is None:
        act_fn = getattr(module, "act", None)
    if all(isinstance(value, nn.Module) for value in (gate_proj, up_proj, down_proj)) and callable(act_fn):
        return gate_proj, up_proj, down_proj, act_fn
    return None


def _straight_through_value(value: torch.Tensor, surrogate: torch.Tensor) -> torch.Tensor:
    """Return ``value`` in the forward pass and ``surrogate``'s derivative."""
    return value.detach() + (surrogate - surrogate.detach())


def _activation_identity_rule(value: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Preserve f(x), using f(x)/stabilize(x) as the detached RelP multiplier.

    Identity refers to relevance, not a unit derivative. Match the reference
    sign-preserving 1e-6 stabilizer (positive at zero); form the ratio in FP32
    for low-precision inputs and retain FP64 for double-precision callers.
    """
    work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    detached = x.detach().to(work_dtype)
    denominator = detached + ((detached == 0).to(work_dtype) + detached.sign()) * 1e-6
    multiplier = value.detach().to(work_dtype) / denominator
    surrogate = (x.to(work_dtype) * multiplier).to(value.dtype)
    return _straight_through_value(value, surrogate)


class _IdentityHalfActivation(nn.Module):
    """Preserve an activation module's forward while applying RelP Identity/Half backward."""

    def __init__(self, activation: nn.Module):
        super().__init__()
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.activation(x)
        identity = _activation_identity_rule(value, x)
        # This activation is one input to Gemma-4's multiplicative per-layer
        # input gate.  Half-rule assigns half of the product relevance to this
        # residual-dependent branch.
        if identity.requires_grad:
            identity.register_hook(lambda gradient: gradient * 0.5)
        return identity


class RelPBackwardRules(AbstractContextManager):
    """Temporarily install the Gemma-compatible RelP backward rule set.

    The edit is injected at the output of ``runtime.layer_module``.  Only
    modules downstream of that edit point are modified, because upstream rules
    cannot affect relevance propagated back to the edit tensor.
    """

    def __init__(self, runtime):
        self.runtime = runtime
        self._patched: list[tuple[nn.Module, object]] = []
        self._patched_attributes: list[tuple[object, str, object]] = []
        self.diagnostics: RelPRuleDiagnostics | None = None

    def _patch_rmsnorm(self, module: nn.Module) -> None:
        original = module.forward

        def forward(this, hidden_states, *args, **kwargs):
            # Evaluate the module's exact native forward without recording its
            # ordinary normalization Jacobian.  For RMSNorm, LN-rule treats the
            # input-dependent normalization scale as constant. Compute it
            # directly: output/input loses precision and is undefined at zero.
            with torch.no_grad():
                value = original(hidden_states, *args, **kwargs)
            x = hidden_states
            if type(this).__name__ == "Qwen3RMSNorm":
                # Qwen3 uses direct weights. Compute the
                # frozen scale directly, including at exact-zero coordinates.
                scale = torch.rsqrt(
                    x.detach().float().square().mean(dim=-1, keepdim=True)
                    + this.variance_epsilon
                )
                surrogate = (x.float() * scale).to(x.dtype) * this.weight.detach()
                return _straight_through_value(value, surrogate)
            name = type(this).__name__
            mean_squared = x.detach().float().square().mean(dim=-1, keepdim=True) + this.eps
            # Gemma-4 uses pow(-0.5) and direct weights (or no weight).
            # Earlier Gemma versions use rsqrt and a 1 + weight offset.
            scale = mean_squared.pow(-0.5) if name == "Gemma4RMSNorm" else torch.rsqrt(mean_squared)
            weight = getattr(this, "weight", None)
            if weight is not None:
                weight = weight.detach().float()
                if name in {"GemmaRMSNorm", "Gemma2RMSNorm", "Gemma3RMSNorm"}:
                    weight = 1.0 + weight
                scale = scale * weight
            # Keep the multiplier in FP32 until the surrogate output cast;
            # autograd also multiplies the incoming gradient in FP32.
            surrogate = (x.float() * scale).to(value.dtype)
            return _straight_through_value(value, surrogate)

        self._patched.append((module, original))
        module.forward = MethodType(forward, module)

    def _patch_gated_mlp(self, module: nn.Module) -> None:
        original = module.forward
        parts = _gated_mlp_parts(module)
        if parts is None:
            return
        gate_proj, up_proj, down_proj, act_fn = parts

        def forward(this, hidden_states, *args, **kwargs):
            if args or kwargs:
                # Current Gemma/Qwen MLPs take only hidden_states.  If a custom
                # remote-code model adds semantics here, do not silently alter
                # them: fall back to the native forward for correctness.
                return original(hidden_states, *args, **kwargs)
            gate_pre = gate_proj(hidden_states)
            gate_value = act_fn(gate_pre)
            # Identity of relevance, not a unit activation derivative.
            gate_identity = _activation_identity_rule(gate_value, gate_pre)
            up_value = up_proj(hidden_states)
            product = gate_identity * up_value
            # Half-rule used by the official RelP Gemma2 implementation:
            # preserve the product while halving the backward signal through
            # the multiplicative gate to avoid relevance doubling.
            half_rule = product * 0.5 + (product * 0.5).detach()
            return down_proj(half_rule)

        self._patched.append((module, original))
        module.forward = MethodType(forward, module)

    def _patch_per_layer_input_gate(self, layer: nn.Module) -> bool:
        """Apply Identity + Half rules to Gemma-4 per-layer embeddings.

        Gemma-4 E2B/E4B uses a second multiplicative gate after the ordinary
        MLP: ``act(per_layer_input_gate(h)) * per_layer_input``.  The
        per-layer embedding bypasses the residual edit point, but the gate is
        still a multiplicative node downstream of that point.  Wrapping the
        layer activation preserves its exact forward value, gives it the
        Identity-rule derivative, and halves relevance entering the residual
        branch, which is the edit-relevant half of the Half-rule split.
        """
        gate = getattr(layer, "per_layer_input_gate", None)
        act_fn = getattr(layer, "act_fn", None)
        enabled = bool(getattr(layer, "hidden_size_per_layer_input", 0))
        if not enabled or not isinstance(gate, nn.Module) or not callable(act_fn):
            return False
        original_act = act_fn
        if isinstance(original_act, nn.Module):
            wrapped = _IdentityHalfActivation(original_act)
        else:
            # Some remote-code models expose activations as plain callables.
            # Those attributes are not registered child modules, so a callable
            # replacement is legal.  Gemma-4 uses an nn.Module and therefore
            # takes the wrapper-module branch above.
            def wrapped(x):
                value = original_act(x)
                identity = _activation_identity_rule(value, x)
                if identity.requires_grad:
                    identity.register_hook(lambda gradient: gradient * 0.5)
                return identity

        self._patched_attributes.append((layer, "act_fn", original_act))
        setattr(layer, "act_fn", wrapped)
        return True

    def __enter__(self):
        layers = list(_resolve_layers_without_transformers(self.runtime.model))
        try:
            edit_index = next(
                index for index, layer in enumerate(layers) if layer is self.runtime.layer_module
            )
        except StopIteration as exc:
            raise RuntimeError("RelP could not locate the edit layer in the transformer stack") from exc
        downstream = layers[edit_index + 1 :]
        if getattr(getattr(self.runtime.model, "config", None), "model_type", None) == "qwen3":
            # Validate the actual runtime architecture before changing methods.
            for index, layer in enumerate(downstream, edit_index + 1):
                attention = getattr(layer, "self_attn", None)
                norms = [getattr(layer, name, None) for name in ("input_layernorm", "post_attention_layernorm")]
                norms += [getattr(attention, name, None) for name in ("q_norm", "k_norm")]
                if not all(isinstance(norm, nn.Module) and _is_rmsnorm(norm) for norm in norms):
                    raise RuntimeError(f"RelP Qwen3 layer {index}: expected residual and Q/K RMSNorm modules")
                if _gated_mlp_parts(getattr(layer, "mlp", None)) is None:
                    raise RuntimeError(f"RelP Qwen3 layer {index}: unsupported gated MLP")
            if not _is_rmsnorm(getattr(self.runtime, "final_norm", None)):
                raise RuntimeError("RelP Qwen3: missing final RMSNorm")

        rmsnorms: list[nn.Module] = []
        mlps: list[nn.Module] = []
        per_layer_input_gates = 0
        seen: set[int] = set()
        for layer in downstream:
            if self._patch_per_layer_input_gate(layer):
                per_layer_input_gates += 1
            for module in layer.modules():
                if id(module) in seen:
                    continue
                seen.add(id(module))
                if _is_rmsnorm(module):
                    rmsnorms.append(module)
                elif _gated_mlp_parts(module) is not None:
                    mlps.append(module)
        final_norm = getattr(self.runtime, "final_norm", None)
        if isinstance(final_norm, nn.Module) and id(final_norm) not in seen and _is_rmsnorm(final_norm):
            rmsnorms.append(final_norm)

        if len(downstream) > 0 and not rmsnorms:
            raise RuntimeError(
                "RelP found no downstream RMSNorm modules. The current implementation "
                "is intentionally validated for Gemma-style RMSNorm transformers; "
                "refusing to silently fall back to ordinary gradients."
            )
        # Patch whole gated MLP forwards before norms. They are disjoint in the
        # supported Gemma architecture; ordinary linear backward implements the
        # LRP-0 rule and attention intentionally remains unchanged.
        for module in mlps:
            self._patch_gated_mlp(module)
        for module in rmsnorms:
            self._patch_rmsnorm(module)

        self.diagnostics = RelPRuleDiagnostics(
            downstream_layers=len(downstream),
            rmsnorm_modules=len(rmsnorms),
            gated_mlp_modules=len(mlps),
            per_layer_input_gates=per_layer_input_gates,
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        for module, original in reversed(self._patched):
            module.forward = original
        for obj, name, original in reversed(self._patched_attributes):
            setattr(obj, name, original)
        self._patched.clear()
        self._patched_attributes.clear()
        return False


def multi_scalar_relevance_patching(
    actual: torch.Tensor,
    function,
    *,
    utility_count: int,
    runtime,
) -> list[dict]:
    """RelP contribution of the applied edit for several scalar utilities.

    This mirrors attribution patching at the no-edit baseline: the edit delta
    ``actual - 0`` is multiplied elementwise by the LRP propagation coefficient
    of each utility.  Forward values remain the ordinary model values.
    """
    baseline = torch.zeros_like(actual).detach().requires_grad_(True)
    with RelPBackwardRules(runtime) as rules:
        start_values = function(baseline).reshape(1, int(utility_count))[0]
        with torch.no_grad():
            end_values = function(actual).reshape(1, int(utility_count))[0].float()
        gradients: list[torch.Tensor] = []
        for index in range(int(utility_count)):
            gradient = torch.autograd.grad(
                start_values[index],
                baseline,
                retain_graph=index + 1 < int(utility_count),
                create_graph=False,
            )[0]
            gradients.append(gradient.detach().float())
        diagnostics = rules.diagnostics
    if diagnostics is None:
        raise RuntimeError("RelP rule diagnostics were not initialized")

    rows = []
    for index, gradient in enumerate(gradients):
        attribution = actual.detach().float() * gradient
        rows.append(
            {
                "gradient": gradient,
                "attribution": attribution,
                "start_value": float(start_values[index].detach().cpu()),
                "end_value": float(end_values[index].detach().cpu()),
                "expected_delta": float((end_values[index] - start_values[index].detach()).cpu()),
                "attributed_delta": float(attribution.sum().detach().cpu()),
                "relp_rules": list(diagnostics.rules),
                "relp_attention_rule": diagnostics.attention_rule,
                "relp_downstream_layers": int(diagnostics.downstream_layers),
                "relp_rmsnorm_modules": int(diagnostics.rmsnorm_modules),
                "relp_gated_mlp_modules": int(diagnostics.gated_mlp_modules),
                "relp_per_layer_input_gates": int(diagnostics.per_layer_input_gates),
            }
        )
    return rows
