from __future__ import annotations

"""Production activation edit modules and shared training utilities."""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class _InputNormMixin:
    def _init_input_norm(self, d_model: int) -> None:
        self.input_norm_mode = "rms"

    def set_input_norm(self, mode: str) -> None:
        if str(mode) != "rms":
            raise ValueError("Only per-token RMS input normalization is supported")
        self.input_norm_mode = "rms"

    def preprocess_input(self, h: torch.Tensor) -> torch.Tensor:
        x = h.float()
        rms = x.square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        return x / rms


class SteeringVectorEdit(nn.Module):
    """Input-independent signed residual direction sanity check."""

    def __init__(self, d_model: int):
        super().__init__()
        self.v = nn.Parameter(torch.zeros(d_model))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.v.to(device=h.device, dtype=h.dtype).view(1, 1, -1).expand_as(h)

    def diagnostics(self) -> dict:
        return {"steering_norm": float(self.v.detach().float().norm().cpu())}


class BaseMLPEdit(_InputNormMixin, nn.Module):
    """Standard per-token SwiGLU residual editor.

    Production uses hidden width equal to the model residual/embed dimension,
    a SiLU-gated GLU feature map, shared gate/up projections across directions,
    and independent directional residual-write heads. Gate/up use standard
    He-normal initialization while the directional write heads start at exact
    zero. The write-head zero initialization preserves a no-edit initial
    function; gate/up cannot also be zero initialized because an all-zero gated
    MLP has zero gradient everywhere and never learns.
    """

    def __init__(
        self,
        d_model: int,
        intermediate_size: int | None = None,
        *,
        activation_name: str = "silu",
        gated: bool = True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        resolved_width = self.d_model if intermediate_size is None else int(intermediate_size)
        if resolved_width <= 0:
            raise ValueError("intermediate_size must be positive")
        self.hidden_dim = resolved_width
        self.intermediate_size = resolved_width
        self.activation_name = str(activation_name)
        self.gated = bool(gated)
        self._init_input_norm(d_model)

        self.gate_proj = nn.Linear(self.d_model, self.intermediate_size, bias=False)
        self.up_proj = (
            nn.Linear(self.d_model, self.intermediate_size, bias=False)
            if self.gated
            else None
        )
        self.output_heads = nn.ModuleDict(
            {
                "c_to_i": nn.Linear(self.intermediate_size, self.d_model, bias=False),
                "i_to_c": nn.Linear(self.intermediate_size, self.d_model, bias=False),
            }
        )

        with torch.no_grad():
            # Standard He-normal initialization for the learned upstream
            # feature projections. ``relu`` selects the canonical sqrt(2/fan_in)
            # gain; the actual forward activation remains the model's configured
            # activation (e.g. SiLU).
            nn.init.kaiming_normal_(
                self.gate_proj.weight, mode="fan_in", nonlinearity="relu"
            )
            if self.up_proj is not None:
                nn.init.kaiming_normal_(
                    self.up_proj.weight, mode="fan_in", nonlinearity="relu"
                )
            # Keep the residual branch exactly disabled at initialization.
            # Once the write heads move away from zero, gradients propagate to
            # the randomly initialized gate/up feature projections.
            for head in self.output_heads.values():
                nn.init.zeros_(head.weight)

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        name = self.activation_name.lower()
        if name in {"silu", "swish"}:
            return F.silu(x)
        if name in {"gelu", "gelu_python"}:
            return F.gelu(x)
        if name in {"gelu_pytorch_tanh", "gelu_new", "gelu_fast"}:
            return F.gelu(x, approximate="tanh")
        if name == "relu":
            return F.relu(x)
        raise ValueError(f"unsupported edit-module activation {self.activation_name!r}")

    def features(self, h: torch.Tensor) -> torch.Tensor:
        x = self.preprocess_input(h).float()
        gate_pre = self.gate_proj(x)
        gated = self._activation(gate_pre)
        if self.up_proj is not None:
            gated = gated * self.up_proj(x)
        return gated

    def decode_features(
        self, features: torch.Tensor, direction: int | torch.Tensor
    ) -> torch.Tensor:
        sign = torch.as_tensor(
            direction, device=features.device, dtype=features.dtype
        ).reshape(-1)
        if sign.numel() == 1:
            sign = sign.expand(int(features.shape[0]))
        if sign.numel() != int(features.shape[0]):
            raise ValueError("direction count must equal the edit batch size")
        forward = self.output_heads["c_to_i"](features.float())
        reverse = self.output_heads["i_to_c"](features.float())
        return torch.where((sign > 0).view(-1, 1, 1), forward, reverse)

    def forward_direction(
        self, h: torch.Tensor, direction: int | torch.Tensor
    ) -> torch.Tensor:
        return self.decode_features(self.features(h), direction).to(dtype=h.dtype)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("BaseMLPEdit uses direction-specific heads; call forward_direction")

    def diagnostics(self) -> dict:
        return {
            "edit_module_intermediate_size": int(self.intermediate_size),
            "edit_module_activation": self.activation_name,
            "edit_module_gated": self.gated,
            "gate_projection_norm": float(
                self.gate_proj.weight.detach().float().norm().cpu()
            ),
            "up_projection_norm": (
                float(self.up_proj.weight.detach().float().norm().cpu())
                if self.up_proj is not None
                else None
            ),
            "c_to_i_output_norm": float(
                self.output_heads["c_to_i"].weight.detach().float().norm().cpu()
            ),
            "i_to_c_output_norm": float(
                self.output_heads["i_to_c"].weight.detach().float().norm().cpu()
            ),
        }


class DirectionalLinearEdit(_InputNormMixin, nn.Module):
    """Full-rank per-token linear sanity check with independent directions."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = int(d_model)
        self.hidden_dim = int(d_model)
        self._init_input_norm(d_model)
        self.output_heads = nn.ModuleDict(
            {
                "c_to_i": nn.Linear(d_model, d_model, bias=False),
                "i_to_c": nn.Linear(d_model, d_model, bias=False),
            }
        )
        with torch.no_grad():
            for head in self.output_heads.values():
                nn.init.zeros_(head.weight)

    def forward_direction(
        self, h: torch.Tensor, direction: int | torch.Tensor
    ) -> torch.Tensor:
        x = self.preprocess_input(h).float()
        sign = torch.as_tensor(direction, device=x.device, dtype=x.dtype).reshape(-1)
        if sign.numel() == 1:
            sign = sign.expand(int(x.shape[0]))
        if sign.numel() != int(x.shape[0]):
            raise ValueError("direction count must equal the edit batch size")
        forward = self.output_heads["c_to_i"](x)
        reverse = self.output_heads["i_to_c"](x)
        return torch.where((sign > 0).view(-1, 1, 1), forward, reverse).to(h.dtype)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("DirectionalLinearEdit requires forward_direction")

    def diagnostics(self) -> dict:
        return {
            f"{name}_weight_norm": float(head.weight.detach().float().norm().cpu())
            for name, head in self.output_heads.items()
        }


def make_edit_module(
    kind: str,
    d_model: int,
    *,
    intermediate_size: int | None = None,
    activation_name: str = "silu",
    gated_mlp: bool = True,
) -> nn.Module:
    """Construct one of the three current production editors."""
    kind = str(kind)
    if kind == "fixed_steering":
        return SteeringVectorEdit(d_model)
    if kind == "directional_linear":
        return DirectionalLinearEdit(d_model)
    if kind == "base_mlp":
        resolved_width = int(d_model) if intermediate_size is None else int(intermediate_size)
        if resolved_width <= 0:
            raise ValueError("base_mlp requires a positive intermediate_size")
        return BaseMLPEdit(
            d_model,
            resolved_width,
            activation_name=activation_name,
            gated=gated_mlp,
        )
    raise ValueError(
        f"Unknown edit module kind {kind!r}; expected fixed_steering, "
        "directional_linear, or base_mlp"
    )


def _iter_modules(edit_module: nn.Module):
    if isinstance(edit_module, nn.ModuleDict):
        yield "control_to_intervention", edit_module["control_to_intervention"]
        yield "intervention_to_control", edit_module["intervention_to_control"]
    else:
        yield "model", edit_module


def set_edit_input_norm(edit_module, mode: str) -> None:
    if str(mode) != "rms":
        raise ValueError("Only per-token RMS input normalization is supported")
    for _name, module in _iter_modules(edit_module):
        if hasattr(module, "set_input_norm"):
            module.set_input_norm(mode)


class EditPatchHook:
    """Add a direction-conditioned residual edit to one Transformer layer."""

    def __init__(
        self,
        layer_module,
        edit_module: nn.Module,
        direction: int | torch.Tensor,
        position_mask: torch.Tensor | None = None,
    ):
        self.layer_module = layer_module
        self.edit_module = edit_module
        self.direction = direction
        self.position_mask = position_mask
        self.handle = None

    def _hook(self, _module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        is_list = isinstance(output, list)
        hidden = output[0] if (is_tuple or is_list) else output
        if hasattr(self.edit_module, "forward_direction"):
            signed = self.edit_module.forward_direction(
                hidden, self.direction
            ).to(device=hidden.device, dtype=hidden.dtype)
        else:
            edit = self.edit_module(hidden)
            if isinstance(self.direction, torch.Tensor):
                direction = self.direction.to(
                    device=hidden.device, dtype=hidden.dtype
                ).view(-1, 1, 1)
            else:
                direction = torch.as_tensor(
                    float(self.direction),
                    device=hidden.device,
                    dtype=hidden.dtype,
                ).view(1, 1, 1)
            signed = direction * edit.to(device=hidden.device, dtype=hidden.dtype)
        if self.position_mask is not None:
            mask = self.position_mask.to(device=hidden.device, dtype=hidden.dtype)
            if mask.shape != hidden.shape[:2]:
                raise ValueError(
                    "edit position mask must match the hidden-state batch and sequence dimensions"
                )
            signed = signed * mask.unsqueeze(-1)
        patched = hidden + signed
        if is_tuple:
            return (patched, *output[1:])
        if is_list:
            out = list(output)
            out[0] = patched
            return out
        return patched

    def __enter__(self):
        self.handle = self.layer_module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def save_checkpoint(
    path: Path, edit_module: nn.Module, *, config: dict, metrics: dict
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": edit_module.state_dict(),
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def num_update_steps(
    num_examples: int, *, batch_size: int, grad_accum_steps: int
) -> int:
    micro_batches = int(
        math.ceil(max(0, int(num_examples)) / max(1, int(batch_size)))
    )
    return int(math.ceil(micro_batches / max(1, int(grad_accum_steps))))
