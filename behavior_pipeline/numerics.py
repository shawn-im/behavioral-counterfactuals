"""Fail explicitly on invalid numerical results instead of repairing them."""

import torch


def reject_invalid(values: torch.Tensor, invalid: torch.Tensor, name: str) -> None:
    if bool(invalid.any()):
        index = tuple(invalid.nonzero()[0].tolist())
        raise RuntimeError(f"{name}: invalid value {values[index].item()} at index {index}")


def require_finite_gradients(named_parameters) -> None:
    for name, parameter in named_parameters:
        if parameter.grad is not None:
            grad = parameter.grad.detach()
            if grad.is_sparse:
                grad = grad.coalesce().values()
            reject_invalid(grad, ~torch.isfinite(grad), f"gradient for {name}")


def validate_kl(values: torch.Tensor) -> torch.Tensor:
    # FP32 normalization/subtraction can produce tiny negative roundoff.
    reject_invalid(values, ~torch.isfinite(values) | (values < -1e-5), "conditional KL")
    return values
